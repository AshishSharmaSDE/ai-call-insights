from fastapi import APIRouter
from pydantic import BaseModel
from app.services import sentiment_service

router = APIRouter(prefix="/api", tags=["Sentiment"])

class SentimentInput(BaseModel):
    text: str

@router.post("/sentiment")
async def analyze_sentiment(payload: SentimentInput):
    sentiment = sentiment_service.analyze_sentiment(payload.text)
    return {"text": payload.text, "sentiment": sentiment}
