from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime

class ExplanationBase(BaseModel):
    method: str = Field(..., description="Explanation method", example="SHAP")
    explanation_data: dict = Field(..., description="Explanation data as a dictionary", example={"feature": "amount", "value": 100})

class ExplanationCreate(ExplanationBase):
    transaction_id: int = Field(..., description="ID of the related transaction", example=1)

class ExplanationRead(ExplanationBase):
    id: int = Field(..., example=1)
    transaction_id: int = Field(..., example=1)
    created_at: datetime = Field(..., example="2024-06-01T12:00:00Z")

    class Config:
        orm_mode = True

class AnomalyBase(BaseModel):
    anomaly_score: float = Field(..., description="Anomaly score", example=0.95)
    description: str = Field(..., description="Description of the anomaly", example="Unusually high transaction")

class AnomalyCreate(AnomalyBase):
    transaction_id: int = Field(..., description="ID of the related transaction", example=1)

class AnomalyRead(AnomalyBase):
    id: int = Field(..., example=1)
    transaction_id: int = Field(..., example=1)
    created_at: datetime = Field(..., example="2024-06-01T12:00:00Z")

    class Config:
        orm_mode = True

class FeedbackBase(BaseModel):
    feedback_text: str = Field(..., description="User feedback text", example="This is not an anomaly.")
    is_anomaly_correct: bool = Field(..., description="Whether the anomaly detection was correct", example=True)

class FeedbackCreate(FeedbackBase):
    transaction_id: int = Field(..., description="ID of the related transaction", example=1)

class FeedbackRead(FeedbackBase):
    id: int = Field(..., example=1)
    transaction_id: int = Field(..., example=1)
    created_at: datetime = Field(..., example="2024-06-01T12:00:00Z")

    class Config:
        orm_mode = True

class TransactionBase(BaseModel):
    user_id: int = Field(..., description="ID of the user", example=1)
    amount: float = Field(..., gt=0, description="Transaction amount, must be positive", example=150.0)
    category: str = Field(..., min_length=1, description="Transaction category", example="Groceries")

class TransactionCreate(TransactionBase):
    @validator('category')
    def category_whitelist(cls, v):
        allowed = {"Groceries", "Utilities", "Rent", "Dining", "Travel"}
        if v not in allowed:
            raise ValueError(f"Category '{v}' is not allowed. Allowed categories: {allowed}")
        return v

class TransactionRead(TransactionBase):
    id: int = Field(..., example=1)
    timestamp: datetime = Field(..., example="2024-06-01T12:00:00Z")
    explanations: List[ExplanationRead] = Field(default_factory=list, description="List of explanations")
    anomalies: List[AnomalyRead] = Field(default_factory=list, description="List of anomalies")
    feedbacks: List[FeedbackRead] = Field(default_factory=list, description="List of feedbacks")

    class Config:
        orm_mode = True

