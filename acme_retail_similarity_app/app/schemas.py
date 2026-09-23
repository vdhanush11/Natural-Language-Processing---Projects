from typing import Optional, Literal

from pydantic import BaseModel, Field


ModelName = Literal["TF-IDF", "Word2Vec", "FastText"]


class RecommendationRequest(BaseModel):
    query_text: Optional[str] = Field(default=None, max_length=1000)
    catalog_id: Optional[int] = None
    model: ModelName = "FastText"
    top_n: int = Field(default=10, ge=1, le=50)
    main_category: Optional[str] = None
    sub_category: Optional[str] = None


class CompareRequest(BaseModel):
    query_text: Optional[str] = Field(default=None, max_length=1000)
    catalog_id: Optional[int] = None
    top_n: int = Field(default=10, ge=1, le=20)


class TrainRequest(BaseModel):
    model: Literal["TF-IDF", "Word2Vec", "FastText", "all"] = "all"
