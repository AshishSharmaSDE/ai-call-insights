# app/services/realtime_service.py
import asyncio
import uuid
import logging
import numpy as np
from typing import Dict, Optional
from collections import deque
from app.services.whisper_service import transcribe_chunk_bytes_async
from app.services.sentiment_service import analyze_sentiment

# Logging setup
logger = logging.getLogger("app.services.realtime_service")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

_sessions: Dict[str, "Session"] = {}

# --- Tunables ---
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
SILENCE_THRESHOLD = 130        # RMS below this = silence
MIN_SILENCE_DURATION = 1.2     # Seconds of silence before flush
MIN_BUFFER_SEC = 4             # Minimum audio before flush
MAX_BUFFER_SEC = 30            # Hard cap per flush
MIN_CHUNK_FOR_ANALYSIS = 40000


class Session:
    def __init__(self, session_id: str, websocket):
        self.id = session_id
        self.websocket = websocket
        self.queue: asyncio.Queue = asyncio.Queue()
        self.task: Optional[asyncio.Task] = None
        self._closed = False
        self._chunk_counter = 0
        self._buffer = bytearray()
        self._batch_buffer = []  # store batches for Whisper multi-chunk mode
        self._last_flush_time = asyncio.get_event_loop().time()

    async def start(self):
        logger.info(f"[QUEUE:{self.id}] Starting consumer task")
        self.task = asyncio.create_task(self._consumer())

    async def enqueue(self, chunk: bytes):
        if self._closed:
            return
        await self.queue.put(chunk)
        logger.debug(f"[QUEUE:{self.id}] Enqueued chunk ({len(chunk)} bytes)")

    async def close(self):
        if self._closed:
            return
        self._closed = True
        await self.queue.put(None)
        if self.task:
            try:
                await asyncio.wait_for(self.task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(f"[QUEUE:{self.id}] Consumer timeout; cancelling task")
                self.task.cancel()

    async def _consumer(self):
        """
        Buffers incoming audio and flushes intelligently on silence/time limits.
        """
        try:
            logger.debug(f"[QUEUE:{self.id}] Consumer started")
            silence_counter = 0.0
            last_chunk_time = asyncio.get_event_loop().time()

            while True:
                chunk = await self.queue.get()
                if chunk is None:
                    logger.debug(f"[QUEUE:{self.id}] Received sentinel → ending consumer")
                    break

                now = asyncio.get_event_loop().time()
                self._buffer.extend(chunk)

                # --- Compute loudness ---
                try:
                    pcm = np.frombuffer(chunk, dtype=np.int16)
                    rms = float(np.sqrt(np.mean(np.square(pcm)))) if pcm.size > 0 else 0
                except Exception:
                    rms = 0

                # --- Silence tracking ---
                if rms < SILENCE_THRESHOLD:
                    silence_counter += (now - last_chunk_time)
                else:
                    silence_counter = 0.0
                last_chunk_time = now

                buffer_seconds = len(self._buffer) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
                time_since_flush = now - self._last_flush_time

                # --- Flush conditions ---
                if buffer_seconds >= MAX_BUFFER_SEC:
                    logger.info(f"[QUEUE:{self.id}] Max buffer ({buffer_seconds:.1f}s) reached — forcing flush")
                    await self._flush_buffer()
                    self._last_flush_time = now

                elif silence_counter >= MIN_SILENCE_DURATION and buffer_seconds >= MIN_BUFFER_SEC:
                    logger.info(f"[QUEUE:{self.id}] Detected pause ({silence_counter:.1f}s) — flushing {buffer_seconds:.1f}s")
                    await self._flush_buffer()
                    self._last_flush_time = now
                    silence_counter = 0.0

            # --- Final flush on session close ---
            if len(self._buffer) > MIN_CHUNK_FOR_ANALYSIS:
                logger.info(f"[QUEUE:{self.id}] Final flush ({len(self._buffer)} bytes)")
                await self._flush_buffer()

            logger.info(f"[QUEUE:{self.id}] Consumer finished")

        except asyncio.CancelledError:
            logger.info(f"[QUEUE:{self.id}] Consumer cancelled")
        except Exception as e:
            logger.exception(f"[QUEUE:{self.id}] Consumer error: {e}")

    async def _flush_buffer(self):
        """Flush buffer → transcribe with Whisper → analyze sentiment → send to client."""
        if not self._buffer:
            logger.debug(f"[QUEUE:{self.id}] Nothing to flush, skipping")
            return

        buffer_copy = bytes(self._buffer)
        self._buffer.clear()
        self._chunk_counter += 1
        chunk_no = self._chunk_counter

        # Store for batch context (merging small chunks)
        self._batch_buffer.append(buffer_copy)

        # Only send batch when we have enough accumulated or final flush
        if len(self._batch_buffer) < 2 and chunk_no > 1:
            logger.debug(f"[QUEUE:{self.id}] Accumulating chunk #{chunk_no}, buffer={len(self._batch_buffer)}")
            return

        logger.info(f"[QUEUE:{self.id}] Preparing Whisper transcription for batch of {len(self._batch_buffer)} chunks")

        # Merge batch
        merged_audio = b"".join(self._batch_buffer)
        self._batch_buffer.clear()

        transcript = ""
        sentiment = "Neutral"

        try:
            transcript = await transcribe_chunk_bytes_async(merged_audio, session_id=self.id, chunk_no=chunk_no)
            if transcript.strip():
                sentiment = analyze_sentiment(transcript)
        except Exception as e:
            logger.error(f"[QUEUE:{self.id}] Error during flush: {e}")

        # Send to client
        try:
            if self.websocket.client_state.value == 1:
                await self.websocket.send_json({"transcript": transcript, "sentiment": sentiment})
                logger.info(f"[QUEUE:{self.id}] Sent transcript chunk #{chunk_no} ({len(merged_audio)} bytes)")
        except Exception as e:
            logger.debug(f"[QUEUE:{self.id}] Failed to send WebSocket message: {e}")


# --- Public API ---
async def create_session(session_id: str, websocket) -> str:
    if session_id in _sessions:
        await stop_session(session_id)
    s = Session(session_id, websocket)
    _sessions[session_id] = s
    await s.start()
    return session_id


async def enqueue_chunk(session_id: str, chunk: bytes):
    s = _sessions.get(session_id)
    if not s:
        raise RuntimeError("Session not found")
    await s.enqueue(chunk)


async def stop_session(session_id: str):
    s = _sessions.pop(session_id, None)
    if s:
        await s.close()
