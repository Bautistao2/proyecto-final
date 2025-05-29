from fastapi import APIRouter, Depends, HTTPException, status
import numpy as np
import logging
from typing import Annotated

from src.schemas.prediction import PredictionRequest, PredictionResponse
from src.services.model_service import ModelService
from src.services.database_service import db_service
from src.core.dependencies import get_model_service

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/predict", 
    response_model=PredictionResponse,
    summary="Realizar predicción de eneagrama",
    description="Predice el tipo de eneagrama y ala basado en las respuestas del test"
)
async def predict_enneagram(
    request: PredictionRequest,
    model_service: Annotated[ModelService, Depends(get_model_service)]
) -> PredictionResponse:
    """
    Realiza una predicción del tipo de eneagrama basada en las respuestas proporcionadas.
    
    - Las respuestas deben ser una lista de 80 números enteros entre 1 y 5
    - Devuelve el tipo principal, el ala y las probabilidades para cada tipo
    """
    # Validación adicional: asegurar que los valores estén entre 1 y 5
    if any((not isinstance(ans, int)) or ans < 1 or ans > 5 for ans in request.answers):
        logger.error("Las respuestas deben ser enteros entre 1 y 5.")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Las respuestas deben ser enteros entre 1 y 5.")
    try:
        # Convertir respuestas a numpy array
        answers = np.array(request.answers, dtype=np.float32).reshape(1, -1)
        
        # Realizar predicción
        result = await model_service.predict(answers)
        
        # Guardar resultado en base de datos
        session_id = await db_service.save_test_result(
            answers=request.answers,  # Lista original de respuestas
            eneatipo=result["enneagram_type"],  # Mapear correctamente
            ala=result["wing"]  # Mapear correctamente
        )
        
        # Añadir session_id a la respuesta
        if session_id:
            result["session_id"] = session_id
            logger.info(f"✅ Resultado guardado con session_id: {session_id}")
        else:
            logger.warning("⚠️ No se pudo guardar el resultado en base de datos")
        
        return PredictionResponse(**result)
        
    except ValueError as e:
        logger.error(f"Error de validación: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error en la predicción: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error interno del servidor")
