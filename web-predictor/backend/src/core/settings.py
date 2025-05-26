from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models" / "artefacts"
# Ruta alternativa para buscar el modelo en Docker (volumen montado)
DOCKER_MODEL_DIR = Path("/app/results")

# Project settings
PROJECT_NAME = "Enneagram Predictor API"
VERSION = "1.0.0"
DEBUG = True

# Model settings
# Función para encontrar el modelo más reciente en una carpeta
def find_latest_model(directory):
    if not directory.exists():
        return None
    model_files = list(directory.glob("*.joblib"))
    if not model_files:
        return None
    # Ordenar por fecha de modificación, más reciente primero
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    return str(latest_model)

# Buscar primero en el volumen montado, luego en la ruta predeterminada
MODEL_PATH = find_latest_model(DOCKER_MODEL_DIR) or find_latest_model(MODELS_DIR) or str(MODELS_DIR / "model.joblib")
MODEL_VERSION = "20250516"

# API settings
API_V1_PREFIX = "/api/v1"
CORS_ORIGINS = [
    "http://localhost:3000"
]

# Questions settings
QUESTIONS_FILE = BASE_DIR / "static" / "data" / "questions.json"
