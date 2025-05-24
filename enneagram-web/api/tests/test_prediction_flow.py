import sys
import os
import numpy as np
import requests
import json
from pathlib import Path

class TestEnneagramPrediction:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.num_questions = 80
        
    def simulate_answers(self):
        """Simula respuestas de usuario (1-5 escala Likert)"""
        return np.random.randint(1, 6, size=self.num_questions).tolist()
        
    def test_prediction_flow(self):
        print("=== Iniciando prueba de predicción ===")
        
        # 1. Obtener preguntas
        print("\n1. Obteniendo preguntas...")
        try:
            response = requests.get(f"{self.base_url}/api/questions")
            questions = response.json()
            print(f"✓ {len(questions)} preguntas recibidas")
        except Exception as e:
            print(f"✗ Error obteniendo preguntas: {e}")
            return
            
        # 2. Simular respuestas
        print("\n2. Simulando respuestas...")
        answers = self.simulate_answers()
        print("✓ Respuestas generadas:", answers[:5], "...")
        
        # 3. Enviar predicción
        print("\n3. Enviando predicción...")
        try:
            # Cambiar 'responses' por 'answers' en el payload
            payload = {"answers": answers}
            response = requests.post(
                f"{self.base_url}/predict",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                print("\n=== Resultado de predicción ===")
                print(f"Eneatipo: {result['eneatipo']}")
                print(f"Ala: {result['ala']}")
                print("\nProbabilidades de Eneatipos:")
                for i, prob in enumerate(result['probabilidades']['eneatipo']):
                    print(f"Tipo {i+1}: {prob:.3f}")
                    
                print("\nProbabilidades de Alas:")
                for i, prob in enumerate(result['probabilidades']['ala']):
                    print(f"Ala {i+1}: {prob:.3f}")
                    
                # Guardar resultado para análisis
                with open('last_prediction_result.json', 'w') as f:
                    json.dump(result, f, indent=2)
                print("\n✓ Resultado guardado en 'last_prediction_result.json'")
                
            else:
                print(f"\n✗ Error en predicción:")
                print(f"Status: {response.status_code}")
                print(f"Detalle: {response.text}")
                
                # Imprimir información de diagnóstico
                print("\nInformación de diagnóstico:")
                print(f"Longitud de respuestas: {len(answers)}")
                print(f"Rango de respuestas: [{min(answers)}, {max(answers)}]")
                print(f"Headers de respuesta: {dict(response.headers)}")
                
        except Exception as e:
            print(f"✗ Error en request: {e}")

if __name__ == "__main__":
    tester = TestEnneagramPrediction()
    tester.test_prediction_flow()