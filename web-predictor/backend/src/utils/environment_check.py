"""
Utility script to verify the Docker environment and paths
"""
import os
import sys
import logging
from pathlib import Path
import glob

logger = logging.getLogger(__name__)

def check_docker_environment():
    """
    Verifica que el entorno Docker esté correctamente configurado
    """
    # Verificar volúmenes montados
    model_dir = Path("/app/model")
    results_dir = Path("/app/results")
    
    environment_status = {
        "is_docker": os.environ.get("DOCKER", "false").lower() == "true",
        "model_dir_exists": model_dir.exists(),
        "results_dir_exists": results_dir.exists(),
        "python_path": sys.path,
    }
    
    # Verificar si tenemos acceso al módulo del modelo
    try:
        sys.path.append("/app/model")
        import eneagrama_predictor
        environment_status["eneagrama_predictor_importable"] = True
    except ImportError:
        environment_status["eneagrama_predictor_importable"] = False
    
    # Verificar si hay modelos en la carpeta de resultados
    model_files = []
    for possible_dir in [results_dir, model_dir / "results"]:
        if possible_dir.exists():
            model_files.extend(list(possible_dir.glob("*.joblib")))
    
    environment_status["available_models"] = [m.name for m in model_files]
    
    return environment_status

def print_environment_report():
    """
    Imprime un reporte del entorno para diagnóstico
    """
    status = check_docker_environment()
    
    logger.info("=== REPORTE DE ENTORNO ===")
    logger.info(f"En Docker: {status['is_docker']}")
    logger.info(f"Directorio model existe: {status['model_dir_exists']}")
    logger.info(f"Directorio results existe: {status['results_dir_exists']}")
    logger.info(f"Módulo eneagrama_predictor importable: {status['eneagrama_predictor_importable']}")
    
    if status['available_models']:
        logger.info(f"Modelos disponibles: {', '.join(status['available_models'])}")
    else:
        logger.warning("⚠️ No se encontraron modelos .joblib en ninguna ubicación!")
    
    logger.info(f"Python path: {sys.path}")
    logger.info("==========================")
    
    return status
