from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
import os

from src.core.settings import PROJECT_NAME, VERSION, API_V1_PREFIX, CORS_ORIGINS
from src.api.routes import predictor, questions

# Ejecutar verificación de entorno
try:
    from src.utils.environment_check import print_environment_report
    env_status = print_environment_report()
except Exception as e:
    print(f"Error verificando el entorno: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=PROJECT_NAME,
    version=VERSION,
    description="API para el sistema de predicción de eneagramas"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Include routes
app.include_router(predictor.router, prefix=API_V1_PREFIX)
app.include_router(questions.router, prefix=API_V1_PREFIX)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": VERSION}

# Mount the static files directory
try:
    app.mount("/", StaticFiles(directory="public", html=True), name="static")
    logger.info("Static files mounted at /")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
