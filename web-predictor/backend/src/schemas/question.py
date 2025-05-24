from pydantic import BaseModel
from typing import List

class Question(BaseModel):
    id: int
    text: str
    category: str 
    original_id: str | None = None

class QuestionsResponse(BaseModel):
    questions: List[Question]
    total: int
