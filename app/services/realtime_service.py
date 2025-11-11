# app/services/realtime_service.py
import asyncio
import uuid
import logging
import numpy as np
from typing import Dict, Optional
from collections import deque
from app.services.whisper_service import transcribe_chunk_bytes_async
from app.services.sentiment_service import analyze_sentiment

# Logging
logger = logging.getLogger("app.services.realtime_service")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

# Session registry
_sessions: Dict[str, "Session"] = {}

# Tunables
CHUNK_SECONDS = 9
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
CHUNK_SIZE_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE * CHUNK_SECONDS  # ~288 KB for 9s PCM16 @16kHz
SILENCE_THRESHOLD = 100
MIN_SILENCE_DURATION = 1.0
MIN_CHUNK_FOR_ANALYSIS = 48000


class Session:
    def __init__(self, session_id: str, websocket):
        self.id = session_id
        self.websocket = websocket
        self.queue: asyncio.Queue = asyncio.Queue()
        self.task: Optional[asyncio.Task] = None
        self._closed = False
        self._chunk_counter = 0
        self._buffer = bytearray()
        self._silence_window = deque(maxlen=int(SAMPLE_RATE * 1.5))  # ~1.5s audio samples

    async def start(self):
        logger.info(f"[QUEUE:{self.id}] Starting consumer task")
        self.task = asyncio.create_task(self._consumer())

    async def enqueue(self, chunk: bytes):
        if self._closed:
            logger.debug(f"[QUEUE:{self.id}] Dropping chunk because session is closed")
            return
        await self.queue.put(chunk)
        logger.debug(
            f"[QUEUE:{self.id}] Chunk enqueued (size={len(chunk)} bytes); queue_size={self.queue.qsize()}"
        )

    async def close(self):
        if self._closed:
            return
        self._closed = True
        await self.queue.put(None)
        if self.task:
            try:
                await asyncio.wait_for(self.task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(f"[QUEUE:{self.id}] Consumer did not finish in time; cancelling")
                self.task.cancel()

    async def _consumer(self):
        """
        Smart buffered consumer:
        - Accumulates audio until a pause (silence) longer than 2s is detected.
        - Ensures buffer >=5s before flushing to Whisper.
        - Safety flush at 12s if user keeps speaking continuously.
        """
        try:
            logger.debug(f"[QUEUE:{self.id}] Consumer loop started (smart buffer mode)")
            silence_counter = 0
            last_chunk_time = asyncio.get_event_loop().time()

            while True:
                chunk = await self.queue.get()
                if chunk is None:
                    logger.debug(f"[QUEUE:{self.id}] Received sentinel -> finishing consumer")
                    break

                now = asyncio.get_event_loop().time()
                self._buffer.extend(chunk)

                # Convert to PCM for silence analysis
                try:
                    pcm = np.frombuffer(chunk, dtype=np.int16)
                    rms = np.sqrt(np.mean(np.square(pcm)))
                except Exception:
                    rms = 0

                # Detect silence or long speech
                if rms < SILENCE_THRESHOLD:
                    silence_counter += (now - last_chunk_time)
                else:
                    silence_counter = 0  # reset silence timer when voice resumes
                last_chunk_time = now

                buffer_seconds = len(self._buffer) / (SAMPLE_RATE * BYTES_PER_SAMPLE)

                # --- Smart Flush Conditions ---
                if silence_counter >= 2.0 and buffer_seconds >= 5:
                    logger.info(f"[QUEUE:{self.id}] Detected pause ({silence_counter:.1f}s) → flushing {buffer_seconds:.1f}s buffer")
                    await self._flush_buffer()
                    silence_counter = 0
                elif buffer_seconds >= 12:
                    logger.info(f"[QUEUE:{self.id}] Max buffer (12s) reached → flushing")
                    await self._flush_buffer()
                    silence_counter = 0

            # Final flush
            if len(self._buffer) > MIN_CHUNK_FOR_ANALYSIS:
                logger.info(f"[QUEUE:{self.id}] Final flush of {len(self._buffer)} bytes")
                await self._flush_buffer()

            logger.info(f"[QUEUE:{self.id}] Consumer finished")

        except asyncio.CancelledError:
            logger.info(f"[QUEUE:{self.id}] Consumer cancelled")
        except Exception as exc:
            logger.error(f"[QUEUE:{self.id}] Consumer error: {exc}")


    async def _flush_buffer(self):
        """Flush the current buffer → Whisper → Sentiment → WebSocket."""
        if not self._buffer:
            return

        buffer_copy = bytes(self._buffer)
        self._buffer.clear()
        self._silence_window.clear()

        self._chunk_counter += 1
        chunk_no = self._chunk_counter
        transcript = ""
        sentiment = "Neutral"

        try:
            transcript = await transcribe_chunk_bytes_async(
                buffer_copy, session_id=self.id, chunk_no=chunk_no
            )
            if transcript.strip():
                sentiment = analyze_sentiment(transcript)
        except Exception as e:
            logger.error(f"[QUEUE:{self.id}] Flush processing failed: {e}")

        # Send to client
        try:
            if self.websocket.client_state.value == 1:
                await self.websocket.send_json({"transcript": transcript, "sentiment": sentiment})
                logger.info(f"[QUEUE:{self.id}] Sent transcript (#{chunk_no}, len={len(buffer_copy)})")
        except Exception as e:
            logger.debug(f"[QUEUE:{self.id}] Could not send response: {e}")



# --- Public API ---
async def create_session(session_id: str, websocket) -> str:
    if session_id in _sessions:
        logger.warning(f"[QUEUE:{session_id}] Session already exists; recreating")
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
