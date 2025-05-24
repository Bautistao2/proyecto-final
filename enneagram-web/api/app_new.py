import sys
import os
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# Configuración de rutas
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "frontend" / "static"
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = STATIC_DIR / "data"

print(f"Directorio base: {BASE_DIR}")
print(f"Directorio estático: {STATIC_DIR}")
print(f"Directorio frontend: {FRONTEND_DIR}")
print(f"Directorio datos: {DATA_DIR}")

# Crear directorios si no existen
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Inicializar FastAPI
app = FastAPI(title="Enneagram Test API")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Agregar la ruta del módulo eneagrama_predictor al PYTHONPATH
predictor_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'eneagrama_predictor'))
sys.path.append(predictor_path)

from .services.model_service import ModelService

# Configurar el servicio del modelo
MODEL_PATH = Path("C:/Users/bauti/Documents/claude/enneagram-web/models/model_20250516_143648.joblib")
model_service = ModelService(MODEL_PATH)

def get_model_service():
    return model_service

# Definir los modelos de datos
class PredictionRequest(BaseModel):
    answers: List[int]  # Lista de respuestas del 1 al 5

class PredictionResult(BaseModel):
    enneagram_type: int
    wing: int
    type_probabilities: List[float]
    wing_probabilities: List[float]

# Primero definimos todas las rutas API que necesitamos que funcionen
@app.get("/api/questions")
async def get_questions():
    questions_file = DATA_DIR / "questions.json"
    if not questions_file.exists():
        raise HTTPException(status_code=404, detail="Questions file not found")
    try:
        with open(questions_file, "r", encoding="utf-8") as f:
            raw_questions = json.load(f)
            # Ordenar las preguntas por ID antes de extraer el texto
            sorted_questions = sorted(raw_questions, key=lambda x: x["id"])
            # Transformar el array de objetos en un objeto con un array de strings
            questions = {
                "questions": [q["text"] for q in sorted_questions]
            }
            return JSONResponse(content=questions)
    except json.JSONDecodeError as e:
        print(f"Error decodificando JSON: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Invalid JSON format: {str(e)}")
    except Exception as e:
        print(f"Error en get_questions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict_enneagram(
    request: PredictionRequest,
    model_service: ModelService = Depends(get_model_service)
):
    """Realizar una predicción de eneagrama retornando solo tipo y ala"""
    try:
        # Validar datos de entrada
        if len(request.answers) != 80:
            raise HTTPException(
                status_code=400, 
                detail=f"Se esperaban 80 respuestas, se recibieron {len(request.answers)}"
            )
        
        # Convertir respuestas a numpy array
        answers = np.array(request.answers, dtype=np.float32).reshape(1, -1)
        
        # Realizar predicción
        result = await model_service.predict(answers)
        
        # Retornar resultado
        return PredictionResult(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

# Montar los archivos estáticos antes del frontend
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
