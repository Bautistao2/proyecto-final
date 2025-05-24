from pathlib import Path
import numpy as np
import joblib
import logging
import sys
import os
from typing import Dict, Optional
from src.core.settings import MODEL_PATH

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = Path(model_path or MODEL_PATH)
        self.model = self._load_model()
        logger.info(f"Modelo cargado desde: {self.model_path}")
        
    def _load_model(self):
        """Carga el modelo desde el archivo joblib, asegurando que la carpeta eneagrama_predictor esté en sys.path"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"No se encontró el modelo en {self.model_path}")

            # Calcular la ruta absoluta a la carpeta eneagrama_predictor desde la raíz del proyecto
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Sube hasta la raíz del proyecto (ajustar si cambia la estructura)
            project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
            eneagrama_predictor_path = os.path.join(project_root, "eneagrama_predictor")

            logger.info(f"Intentando agregar eneagrama_predictor al sys.path: {eneagrama_predictor_path}")
            logger.info(f"sys.path antes de la inserción: {sys.path}")

            if not os.path.isdir(eneagrama_predictor_path):
                logger.warning(f"La carpeta eneagrama_predictor no existe en: {eneagrama_predictor_path}")
            elif eneagrama_predictor_path not in sys.path:
                sys.path.insert(0, eneagrama_predictor_path)
                logger.info(f"Agregada eneagrama_predictor al sys.path: {eneagrama_predictor_path}")

            logger.info(f"sys.path después de la inserción: {sys.path}")

            # Intentar importar el módulo requerido por el modelo serializado
            try:
                import optimized_enneagram_system
                logger.info("Importación de optimized_enneagram_system exitosa.")
            except ImportError as e:
                logger.error(f"No se pudo importar optimized_enneagram_system: {e}")

            # Cargar el modelo real
            return joblib.load(self.model_path)
        except Exception as e:
            logger.error(f"Error cargando el modelo: {str(e)}")
            raise RuntimeError(f"No se pudo cargar el modelo desde {self.model_path}")

    async def predict(self, answers: np.ndarray) -> Dict:
        """Realiza la predicción usando el modelo cargado"""
        try:
            # Validar forma del input
            if answers.shape[1] != 80:
                raise ValueError("Se esperaban 80 respuestas")

            # Realizar predicción usando el método real del modelo
            predictions = self.model.predict(answers)

            type_probs = predictions['eneatipo_probabilidades']
            wing_probs = predictions['ala_probabilidades']

            return {
                "enneagram_type": int(predictions['eneatipo'][0]),
                "wing": int(predictions['ala'][0]),
                "type_probabilities": type_probs[0].tolist(),
                "wing_probabilities": wing_probs[0].tolist()
            }

        except Exception as e:
            logger.error(f"Error en la predicción: {str(e)}")
            raise
