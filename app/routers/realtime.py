# app/routers/realtime.py
import uuid
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.realtime_service import create_session, enqueue_chunk, stop_session

# logging
logger = logging.getLogger("app.routers.realtime")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)

router = APIRouter(prefix="/api", tags=["Realtime"])


@router.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket):
    """
    WebSocket endpoint:
    - Creates a session object that manages its own queue/task.
    - For each binary frame received from client, enqueues it for processing.
    - Cleans up on disconnect.
    """
    # Create session id
    session_id = uuid.uuid4().hex[:8]
    logger.info(f"[WS:{session_id}] Connection incoming, accepting WebSocket")
    await websocket.accept()
    logger.info(f"[WS:{session_id}] Connection accepted")

    # create backing session in service
    try:
        await create_session(session_id, websocket)
    except Exception as exc:
        logger.error(f"[WS:{session_id}] Failed to create session: {exc}")
        await websocket.close()
        return

    try:
        while True:
            # We expect binary audio frames from client
            try:
                data = await websocket.receive_bytes()
            except Exception as e:
                # could be text frame or connection drop
                text = None
                try:
                    msg = await websocket.receive_text()
                    text = msg
                except Exception:
                    pass

                if text:
                    logger.debug(f"[WS:{session_id}] Received text frame (ignored): {text}")
                    continue
                else:
                    logger.debug(f"[WS:{session_id}] No more frames, breaking")
                    break

            # enqueue the binary audio chunk
            try:
                await enqueue_chunk(session_id, data)
            except Exception as e:
                logger.error(f"[WS:{session_id}] Enqueue failed: {e}")
    except WebSocketDisconnect:
        logger.info(f"[WS:{session_id}] WebSocket disconnected by client")
    except Exception as exc:
        logger.error(f"[WS:{session_id}] Unexpected error: {exc}")
    finally:
        # ensure cleanup
        try:
            await stop_session(session_id)
        except Exception as e:
            logger.error(f"[WS:{session_id}] Error stopping session: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info(f"[WS:{session_id}] Connection closed and session cleaned up")
