from typing import Optional
from pydantic import BaseModel

class DetectRequest(BaseModel):
    text: str
    session_id: Optional[str] = None

class HumanizeRequest(BaseModel):
    text: str

class HumanizeResponse(BaseModel):
    humanized_text: str

class DetectResponse(BaseModel):
    label: str
    confidence: float
    ai_score: float
    human_score: float
    word_count: int
    char_count: int
    analysis_time_ms: float

class DetectFileResponse(DetectResponse):
    filename: str
    extracted_chars: int
