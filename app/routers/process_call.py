from app.models.schemas import ProcessedCallResponse, SentimentResult
from fastapi import APIRouter, UploadFile, File
from app.services import sentiment_service, whisper_service

router = APIRouter(prefix="/api", tags=["Call Processing"])

@router.post("/process-call", response_model=ProcessedCallResponse)
async def process_call(file: UploadFile = File(...)):
    transcript_data = whisper_service.transcribe_audio(file)
    text = transcript_data["text"]
    segments = text.split(". ")

    analysis = [SentimentResult(text=seg, sentiment=sentiment_service.analyze_sentiment(seg))
                for seg in segments if seg.strip()]

    return ProcessedCallResponse(transcript=text, analysis=analysis)
