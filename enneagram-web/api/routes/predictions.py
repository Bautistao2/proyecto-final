from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
from ..services.model_service import ModelService
from pathlib import Path

router = APIRouter()
# Inicializar el servicio sin parámetros para usar la ruta por defecto
model_service = ModelService()

class PredictionRequest(BaseModel):
    answers: List[int]

class PredictionResponse(BaseModel):
    eneatipo: int
    ala: int
    probabilidades: Dict[str, List[float]]
    explicaciones: Dict[str, str]
    grafico_data: Dict[str, List]

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Validate input values
        if not all(isinstance(x, int) and 1 <= x <= 5 for x in request.answers):
            raise HTTPException(
                status_code=400,
                detail="All answers must be integers between 1 and 5"
            )
        if len(request.answers) != 80:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 80 answers, got {len(request.answers)}"
            )
            
        # Convert to numpy array with error handling
        try:
            features = np.array(request.answers, dtype=np.float32)
            features = features.reshape(1, -1)
            print(f"Features shape: {features.shape}, dtype: {features.dtype}")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error converting answers to numpy array: {str(e)}"
            )
        
        # Get prediction with error handling
        try:
            predictions, probabilities, explanations = await model_service.predict(features)
            print(f"Predictions shape: {predictions.shape}")
            print(f"Probabilities keys: {probabilities.keys()}")
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error during prediction: {str(e)}"
            )

        # Validate prediction results and probabilities
        if not all(key in probabilities for key in ['eneatipo', 'ala']):
            raise HTTPException(
                status_code=500,
                detail="Invalid prediction format: missing probability data"
            )        # Convert all numpy arrays to lists in probabilities dictionary
        formatted_probabilities = {
            'eneatipo': probabilities['eneatipo_probabilidades'][0].tolist() if isinstance(probabilities.get('eneatipo_probabilidades'), np.ndarray) else [0] * 9,
            'ala': probabilities['ala_probabilidades'][0].tolist() if isinstance(probabilities.get('ala_probabilidades'), np.ndarray) else [0] * 9
        }
        
        # Ensure probabilities are valid
        for key in ['eneatipo', 'ala']:
            if len(formatted_probabilities[key]) > 9:
                formatted_probabilities[key] = formatted_probabilities[key][:9]
            elif len(formatted_probabilities[key]) < 9:
                formatted_probabilities[key].extend([0] * (9 - len(formatted_probabilities[key])))

        # Create chart data with validation
        try:
            chart_data = {
                'labels': [f'Tipo {i+1}' for i in range(9)],
                'datasets': [{
                    'label': 'Probabilidades de Eneatipos',
                    'data': formatted_probabilities['eneatipo'],
                    'backgroundColor': [
                        'rgba(255, 99, 132, 0.5)',
                        'rgba(54, 162, 235, 0.5)',
                        'rgba(255, 206, 86, 0.5)',
                        'rgba(75, 192, 192, 0.5)',
                        'rgba(153, 102, 255, 0.5)',
                        'rgba(255, 159, 64, 0.5)',
                        'rgba(199, 199, 199, 0.5)',
                        'rgba(83, 102, 255, 0.5)',
                        'rgba(40, 159, 64, 0.5)'
                    ]
                }]
            }
        except Exception as e:
            print(f"Error creating chart data: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error creating chart data: {str(e)}"
            )

        return {
            'eneatipo': int(predictions[0][0]),  # Already adjusted in model service
            'ala': int(predictions[0][1]),       # Already adjusted in model service
            'probabilidades': formatted_probabilities,
            'explicaciones': explanations,
            'grafico_data': chart_data
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in prediction endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
