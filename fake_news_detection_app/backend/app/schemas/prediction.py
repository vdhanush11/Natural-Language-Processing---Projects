from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    title: str = Field(default="", max_length=5000)
    text: str = Field(default="", max_length=50000)


class ExplanationRequest(PredictionRequest):
    top_k: int = Field(default=6, ge=1, le=15)


class Probabilities(BaseModel):
    FAKE: float
    REAL: float


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    probabilities: Probabilities
    cleaned_text_length: int
    model: str


class InfluenceSignal(BaseModel):
    word: str
    impact: float


class ExplanationResponse(BaseModel):
    prediction: PredictionResponse
    influential_words: list[InfluenceSignal]
