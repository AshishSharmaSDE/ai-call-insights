# app/services/whisper_service.py
import asyncio
import tempfile
import os
import logging
import subprocess
from typing import Optional, List, Dict
from io import BytesIO
from pydub import AudioSegment
from pydub.utils import which
import whisper
import difflib

# -------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------
logger = logging.getLogger("app.services.whisper_service")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

# -------------------------------------------------------------
# FFMPEG / Pydub setup
# -------------------------------------------------------------
ffmpeg_path = which("ffmpeg") or "C:\\ffmpeg\\bin\\ffmpeg.exe"
ffprobe_path = which("ffprobe") or "C:\\ffmpeg\\bin\\ffprobe.exe"
AudioSegment.converter = ffmpeg_path
AudioSegment.ffprobe = ffprobe_path
os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)

logger.info(f"[WHISPER] Using ffmpeg: {ffmpeg_path}")
logger.info(f"[WHISPER] Using ffprobe: {ffprobe_path}")

# header cache to repair fragmented WebM
_ffmpeg_header_cache: bytes = b""

# session context for stitching transcripts
_session_last_transcript: Dict[str, str] = {}

# -------------------------------------------------------------
# Whisper model cache
# -------------------------------------------------------------
_MODEL: Optional[whisper.Whisper] = None
# specify through env WHISPER_MODEL_NAME (e.g., "small", "base", "medium")
_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "base")


def _load_model():
    global _MODEL
    if _MODEL is None:
        logger.info(f"[WHISPER] Loading Whisper model '{_MODEL_NAME}' (this may take a while)...")
        _MODEL = whisper.load_model(_MODEL_NAME)
        logger.info("[WHISPER] Model loaded successfully.")
    return _MODEL


# -------------------------------------------------------------
# Helpers: temp file writing
# -------------------------------------------------------------
def _write_bytes_to_tempfile(bytes_data: bytes, suffix: str = ".wav") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(bytes_data)
        tmp.flush()
        tmp.close()
        return tmp.name
    except Exception:
        try:
            tmp.close()
        except Exception:
            pass
        raise


# -------------------------------------------------------------
# FFmpeg conversion (robust)
# -------------------------------------------------------------
def _convert_to_wav_bytes_ffmpeg(audio_bytes: bytes, session_id: str = "default") -> bytes:
    """
    Convert incoming audio bytes (webm/ogg/mp3/raw) to 16kHz mono WAV bytes using ffmpeg.
    - Repairs fragmented WebM by prepending cached EBML header if present.
    - Uses a subprocess with error-tolerant flags.
    """
    global _ffmpeg_header_cache

    if not audio_bytes or len(audio_bytes) < 1200:
        logger.debug(f"[WHISPER:{session_id}] Chunk too small ({len(audio_bytes) if audio_bytes else 0} bytes); skipping")
        return b""

    # If audio_bytes already look like WAV (RIFF header), return as-is
    try:
        if audio_bytes[:4] == b"RIFF":
            logger.debug(f"[WHISPER:{session_id}] Detected WAV bytes directly")
            return audio_bytes
    except Exception:
        pass

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "in.webm")
        output_path = os.path.join(tmpdir, "out.wav")

        # Prepend header if we have cached one and current chunk looks like fragment
        try:
            if _ffmpeg_header_cache and not audio_bytes.startswith(b"\x1a\x45\xdf\xa3"):
                audio_bytes = _ffmpeg_header_cache + audio_bytes
        except Exception:
            pass

        # write input file
        try:
            with open(input_path, "wb") as f:
                f.write(audio_bytes)
        except Exception as e:
            logger.warning(f"[WHISPER:{session_id}] Failed to write temp input: {e}")
            return b""

        # first attempt: regular conversion with normalization
        cmd = [
            ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-i",
            input_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-af",
            "volume=2.0, dynaudnorm",
            output_path,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode(errors="ignore") if e.stderr else ""
            logger.warning(f"[WHISPER:{session_id}] ffmpeg conversion failed (attempt1): {stderr[:300]}")

            # fallback: try piping via stdin (useful for some fragmented data)
            cmd_pipe = [
                ffmpeg_path,
                "-y",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-f",
                "webm",
                "-i",
                "pipe:0",
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-af",
                "volume=2.0, dynaudnorm",
                output_path,
            ]
            try:
                subprocess.run(cmd_pipe, input=audio_bytes, check=True, capture_output=True)
            except subprocess.CalledProcessError as e2:
                stderr2 = e2.stderr.decode(errors="ignore") if e2.stderr else ""
                logger.error(f"[WHISPER:{session_id}] ffmpeg fallback (pipe) failed: {stderr2[:300]}")
                return b""

        # cache EBML header if present
        try:
            if not _ffmpeg_header_cache and audio_bytes.startswith(b"\x1a\x45\xdf\xa3"):
                _ffmpeg_header_cache = audio_bytes[:2048]
                logger.debug(f"[WHISPER:{session_id}] Cached WebM EBML header")
        except Exception:
            pass

        # read output wav
        try:
            with open(output_path, "rb") as f:
                wav_bytes = f.read()
            if not wav_bytes or len(wav_bytes) < 1000:
                logger.debug(f"[WHISPER:{session_id}] Converted WAV too small ({len(wav_bytes) if wav_bytes else 0}), skipping")
                return b""
            return wav_bytes
        except Exception as e:
            logger.warning(f"[WHISPER:{session_id}] Failed to read converted WAV: {e}")
            return b""


# -------------------------------------------------------------
# Merge WAV fragments (if needed)
# -------------------------------------------------------------
def _merge_wav_chunks(chunks: List[bytes]) -> bytes:
    """Merge multiple WAV bytes into a single WAV bytes object using pydub."""
    segments = []
    for c in chunks:
        try:
            seg = AudioSegment.from_file(BytesIO(c), format="wav")
            segments.append(seg)
        except Exception as e:
            logger.warning(f"[WHISPER] Failed to parse WAV chunk for merge: {e}")

    if not segments:
        return b""

    merged = segments[0]
    for seg in segments[1:]:
        merged = merged.append(seg, crossfade=100)  # small crossfade to smooth joins

    out = BytesIO()
    merged.export(out, format="wav")
    return out.getvalue()


# -------------------------------------------------------------
# Transcript stitching (simple overlap removal)
# -------------------------------------------------------------
def _stitch_transcripts(prev: str, curr: str) -> str:
    """
    Simple stitching: if the end of prev overlaps with the start of curr,
    remove the duplicated overlap. Uses word-level comparison.
    """
    if not prev:
        return curr
    if not curr:
        return prev

    prev_words = prev.strip().split()
    curr_words = curr.strip().split()

    # maximum overlap length to check
    max_ol = min(len(prev_words), len(curr_words), 30)  # limit to 30 words overlap
    best_ol = 0
    for ol in range(max_ol, 0, -1):
        if prev_words[-ol:] == curr_words[:ol]:
            best_ol = ol
            break

    if best_ol > 0:
        stitched = " ".join(prev_words + curr_words[best_ol:])
        return stitched
    else:
        # fallback: use difflib to find fuzzy overlap
        sm = difflib.SequenceMatcher(a=prev, b=curr)
        match = sm.find_longest_match(0, len(prev), 0, len(curr))
        if match.size > 20:  # arbitrary threshold of characters
            # remove overlapping substring from curr
            overlap = curr[match.b: match.b + match.size]
            curr_remainder = curr.replace(overlap, "", 1).strip()
            if curr_remainder:
                return (prev + " " + curr_remainder).strip()
        return (prev + " " + curr).strip()


# -------------------------------------------------------------
# Main async transcription API
# -------------------------------------------------------------
async def transcribe_chunk_bytes_async(chunk_bytes: bytes, session_id: Optional[str] = None, chunk_no: Optional[int] = None) -> str:
    """
    Entry point for transcription. Accepts:
      - single WebM fragment or WAV bytes, or
      - merged WAV bytes (from realtime_service)
    Behavior:
      - Convert to WAV if needed,
      - Save to temp file and call Whisper asynchronously,
      - Stitch with previous session transcript to avoid duplication,
      - Return the stitched transcript (and update session context).
    """
    sess_tag = f"[WHISPER:{session_id}]" if session_id else "[WHISPER]"
    try:
        # If bytes look like a concatenation of multiple wavs (e.g. realtime merged),
        # attempt to detect by checking for 'RIFF' occurrences; otherwise convert.
        wav_bytes = None
        if chunk_bytes[:4] == b"RIFF":
            # already WAV
            wav_bytes = chunk_bytes
            logger.debug(f"{sess_tag} Received WAV bytes (size={len(wav_bytes)})")
        else:
            # convert from webm/ogg/opus etc -> wav
            wav_bytes = _convert_to_wav_bytes_ffmpeg(chunk_bytes, session_id=session_id)
            if not wav_bytes:
                logger.debug(f"{sess_tag} Conversion returned empty; skipping transcription")
                return ""

        # Ensure valid wav_bytes
        if not wav_bytes:
            return ""

        # Write out wav and run Whisper in a background thread
        tmp_path = _write_bytes_to_tempfile(wav_bytes, suffix=".wav")
        def blocking_transcribe(path: str) -> str:
            model = _load_model()
            try:
                # ensure we run on CPU fp16=False for safety
                result = model.transcribe(path, fp16=False)
                return result.get("text", "").strip()
            except Exception as ex:
                logger.exception(f"{sess_tag} Blocking transcription error: {ex}")
                raise

        logger.info(f"{sess_tag} Starting transcription (chunk_no={chunk_no})")
        text = await asyncio.to_thread(blocking_transcribe, tmp_path)

        # cleanup
        try:
            os.remove(tmp_path)
        except Exception:
            pass

        if not text:
            logger.debug(f"{sess_tag} Whisper returned empty transcript for chunk_no={chunk_no}")
            return ""

        # Stitch with previous session transcript to avoid duplication/edge-word loss
        stitched = text
        if session_id:
            prev = _session_last_transcript.get(session_id, "")
            if prev:
                stitched = _stitch_transcripts(prev, text)
            _session_last_transcript[session_id] = stitched

        logger.info(f"{sess_tag} Transcription result ({len(text)} chars) chunk_no={chunk_no}: {text[:120]}")
        return stitched

    except Exception as exc:
        logger.error(f"{sess_tag} local whisper transcription failed: {exc}")
        return ""
