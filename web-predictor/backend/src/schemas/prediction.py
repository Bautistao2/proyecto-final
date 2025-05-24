from pydantic import BaseModel, Field
from typing import List

class PredictionRequest(BaseModel):
    answers: List[int] = Field(
        ...,
        description="Lista de respuestas del cuestionario (valores del 1-5)",
        min_items=80,
        max_items=80
    )

class PredictionResponse(BaseModel):
    enneagram_type: int = Field(..., description="Tipo de eneagrama principal (1-9)")
    wing: int = Field(..., description="Ala del eneagrama (1-9)")
    type_probabilities: List[float] = Field(..., description="Probabilidades para cada tipo")
    wing_probabilities: List[float] = Field(..., description="Probabilidades para cada ala")

    class Config:
        schema_extra = {
            "example": {
                "enneagram_type": 5,
                "wing": 4,
                "type_probabilities": [0.1, 0.1, 0.1, 0.15, 0.25, 0.1, 0.1, 0.05, 0.05],
                "wing_probabilities": [0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]
            }
        }
