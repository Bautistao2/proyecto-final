from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models" / "artefacts"

# Project settings
PROJECT_NAME = "Enneagram Predictor API"
VERSION = "1.0.0"
DEBUG = True

# Model settings
MODEL_PATH = str(MODELS_DIR / "model.joblib")
MODEL_VERSION = "20250516"

# API settings
API_V1_PREFIX = "/api/v1"
CORS_ORIGINS = [
    "http://localhost:3000"
]

# Questions settings
QUESTIONS_FILE = BASE_DIR / "static" / "data" / "questions.json"
