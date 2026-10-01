
from typing import List, Optional
from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    model: str = Field(..., description="Simple RNN, LSTM, GRU or DistilGPT-2")
    top_k: int = Field(5, ge=1, le=10)

class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    model: str
    num_words: int = Field(10, ge=1, le=30)

class PredictionItem(BaseModel):
    token: str
    probability: float
    display_probability: str

class PredictionResponse(BaseModel):
    model: str
    input_text: str
    prediction_type: str
    top_predictions: List[PredictionItem]
    predicted_next: str
    generated_text: Optional[str] = None

class GenerateResponse(BaseModel):
    model: str
    input_text: str
    generated_text: str
