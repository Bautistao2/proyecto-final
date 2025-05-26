from pathlib import Path
import numpy as np
import joblib
import logging
import sys
import os
from typing import Dict, Optional
from src.core.settings import MODEL_PATH
from src.services.post_processing import post_process_prediction

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = Path(model_path or MODEL_PATH)
        logger.info(f"Intentando cargar modelo desde: {self.model_path}")
        self.model = self._load_model()
        logger.info(f"Modelo cargado exitosamente desde: {self.model_path}")
        
    def _load_model(self):
        """Carga el modelo desde el archivo joblib, asegurando que la carpeta model/eneagrama_predictor esté en sys.path"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"No se encontró el modelo en {self.model_path}")

            # En Docker, la ruta al módulo del modelo está en /app/model/eneagrama_predictor
            # También intentar buscar en las rutas relativas para desarrollo local
            possible_paths = [
                "/app/model/eneagrama_predictor",  # Ruta en Docker (volumen montado)
                os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../..", "model", "eneagrama_predictor"))  # Ruta desarrollo local
            ]
            
            eneagrama_predictor_path = None
            for path in possible_paths:
                if os.path.isdir(path):
                    eneagrama_predictor_path = path
                    logger.info(f"Encontrada carpeta eneagrama_predictor en: {path}")
                    break
                    
            if not eneagrama_predictor_path:
                logger.warning("No se encontró la carpeta eneagrama_predictor en ninguna ruta posible")

            logger.info(f"Intentando agregar model/eneagrama_predictor al sys.path: {eneagrama_predictor_path}")
            logger.info(f"sys.path antes de la inserción: {sys.path}")

            if not os.path.isdir(eneagrama_predictor_path):
                logger.warning(f"La carpeta model/eneagrama_predictor no existe en: {eneagrama_predictor_path}")
            elif eneagrama_predictor_path not in sys.path:
                sys.path.insert(0, eneagrama_predictor_path)
                logger.info(f"Agregada model/eneagrama_predictor al sys.path: {eneagrama_predictor_path}")

            logger.info(f"sys.path después de la inserción: {sys.path}")

            # Intentar importar el módulo requerido por el modelo serializado
            try:
                import optimized_enneagram_system
                logger.info("Importación de optimized_enneagram_system exitosa.")
            except ImportError as e:
                logger.error(f"No se pudo importar optimized_enneagram_system: {e}")
                
            # Cargar el modelo real
            model = joblib.load(self.model_path)
            
            # Verificar que el modelo sea del tipo correcto
            if not hasattr(model, 'predict') or not callable(model.predict):
                raise TypeError("El modelo cargado no tiene un método predict() válido")
                
            logger.info(f"Modelo cargado exitosamente: {type(model).__name__}")
            return model
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

            raw_prediction = {
                "enneagram_type": int(predictions['eneatipo'][0]),
                "wing": int(predictions['ala'][0]),
                "type_probabilities": type_probs[0].tolist(),
                "wing_probabilities": wing_probs[0].tolist()
            }
            
            # Apply post-processing to ensure wing is adjacent to type
            corrected_prediction = post_process_prediction(raw_prediction)
            
            return corrected_prediction

        except Exception as e:
            logger.error(f"Error en la predicción: {str(e)}")
            raise
