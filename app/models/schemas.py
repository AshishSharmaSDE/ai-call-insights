from pydantic import BaseModel
from typing import List

class SentimentResult(BaseModel):
    text: str
    sentiment: str

class ProcessedCallResponse(BaseModel):
    transcript: str
    analysis: List[SentimentResult]