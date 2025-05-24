from pathlib import Path
import numpy as np
import joblib
from typing import Tuple, Dict
import pandas as pd
import tensorflow as tf
import os

class ModelService:
    def __init__(self, model_path: Path = None):
        # Si no se proporciona una ruta, usar la ruta por defecto
        if model_path is None:
            model_path = Path("enneagram-web/models/model_20250516_143648.joblib")
        
        self.model_path = Path(model_path).resolve()  # Convertir a ruta absoluta
        self.model = None
        self._load_model()

    def _load_model(self):
        """Carga el modelo serializado"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"No se encontró el modelo en {self.model_path}")

            # Cargar el modelo usando ruta absoluta
            self.model = joblib.load(str(self.model_path))
            print(f"Modelo cargado exitosamente desde {self.model_path}")
            print(f"Tipo de modelo cargado: {type(self.model)}")
        except Exception as e:
            print(f"Error cargando el modelo: {str(e)}")
            raise

    async def predict(self, features: np.ndarray) -> Dict:
        """
        Realiza una predicción con el modelo retornando tipo, ala y probabilidades
        Returns:
            Dict: Diccionario con enneagram_type, wing y sus probabilidades
        """
        if self.model is None:
            raise ValueError("El modelo no está cargado")

        try:
            # Asegurar forma correcta del input
            features = features.reshape(1, -1)
            
            # Realizar predicción
            prediction_dict = self.model.predict(features)
            
            # Obtener predicciones y probabilidades
            return {
                'enneagram_type': int(prediction_dict['eneatipo'][0]),
                'wing': int(prediction_dict['ala'][0]),
                'type_probabilities': prediction_dict['eneatipo_probabilidades'].flatten().tolist(),
                'wing_probabilities': prediction_dict['ala_probabilidades'].flatten().tolist()
            }
        except Exception as e:
            print(f"Error en predicción: {str(e)}")
            print(f"Shape de features: {features.shape}")
            if 'prediction_dict' in locals():
                print("Contenido de prediction_dict:")
                for k, v in prediction_dict.items():
                    print(f"{k}: {type(v)}")
            raise

    def _get_type_description(self, type_num: int) -> str:
        """Retorna la descripción del eneatipo"""
        descriptions = {
            1: "Perfeccionista - Motivados por hacer lo correcto y mejorar las cosas.",
            2: "Ayudador - Motivados por el deseo de ser amados y ayudar a otros.",
            3: "Triunfador - Motivados por el éxito y la imagen personal.",
            4: "Individualista - Motivados por la autenticidad y expresión personal.",
            5: "Investigador - Motivados por el conocimiento y la comprensión.",
            6: "Leal - Motivados por la seguridad y el apoyo mutuo.",
            7: "Entusiasta - Motivados por la experiencia y la positividad.",
            8: "Desafiador - Motivados por la fuerza y la independencia.",
            9: "Pacificador - Motivados por la armonía y la paz interior."
        }
        return descriptions.get(type_num, "Tipo desconocido")

    def _get_wing_description(self, wing_num: int) -> str:
        """Retorna la descripción del ala"""
        return f"Ala {wing_num}"
