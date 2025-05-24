from pydantic import BaseModel, Field, conlist
from typing import List, Dict, Annotated

class Question(BaseModel):
    """Modelo para una pregunta del test de eneagrama"""
    id: int = Field(description="Identificador único de la pregunta")
    text: str = Field(description="Texto de la pregunta")
    enneagram_type: int = Field(ge=1, le=9, description="Tipo de eneagrama asociado (1-9)")

class TestResponse(BaseModel):
    """Modelo para las respuestas del test de eneagrama"""
    features: conlist(
        Annotated[int, Field(ge=1, le=5)], 
        min_length=80, 
        max_length=80
    ) = Field(description="Lista de 80 respuestas, cada una con valor entre 1 y 5")

class PredictionResult(BaseModel):
    """Modelo para el resultado de la predicción"""
    enneagram_type: int = Field(
        ge=1, le=9,
        description="Tipo de eneagrama predicho (1-9)"
    )
    wing: int = Field(
        ge=1, le=9,
        description="Ala predicha (1-9)"
    )
    probabilities: Dict[str, float] | None = Field(
        default=None,
        description="Probabilidades de cada tipo (opcional)"
    )
