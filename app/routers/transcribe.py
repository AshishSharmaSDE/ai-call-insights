from fastapi import APIRouter, UploadFile, File
from app.services import whisper_service

router = APIRouter(prefix="/api", tags=["Transcription"])

@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    text = whisper_service.transcribe_audio(file)
    return {"transcript": text}
