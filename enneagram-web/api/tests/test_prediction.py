import requests
import numpy as np

# URL del endpoint
url = "http://localhost:8000/predict"  # Sin el prefijo /api/ ya que lo quitamos del servidor


# Verificar que el servidor está corriendo
def check_server():
    try:
        response = requests.get("http://localhost:8000/api/questions")
        return response.status_code == 200
    except:
        return False

# Esperar a que el servidor esté disponible (máximo 10 segundos)
import time
max_retries = 10
for i in range(max_retries):
    if check_server():
        print("Servidor detectado y funcionando")
        break
    if i < max_retries - 1:
        print(f"Esperando al servidor... intento {i+1}/{max_retries}")
        time.sleep(1)
    else:
        print("No se pudo conectar al servidor. Asegúrate de que esté corriendo con 'uvicorn app_new:app --reload'")
        exit(1)

# Crear un array de 80 respuestas aleatorias entre 1 y 5
test_answers = np.random.randint(1, 6, size=80).tolist()

# Crear el payload
payload = {
    "answers": test_answers
}

# Hacer la petición
try:
    print(f"\nEnviando petición a: {url}")
    print(f"Payload: {len(payload['answers'])} respuestas")
    print("Primeras 5 respuestas como muestra:", test_answers[:5])
    
    response = requests.post(url, json=payload)
    
    print(f"Status Code: {response.status_code}")
    print(f"Headers: {response.headers}")
    
    if response.status_code == 200:
        result = response.json()
        print("\nPredicción exitosa!")
        print(f"Eneatipo: {result['enneagram_type']}")
        print(f"Ala: {result['wing']}")
        print("\nProbabilidades por tipo:")
        for tipo, prob in result['probabilities'].items():
            print(f"Tipo {tipo}: {prob:.2%}")
    else:
        print(f"\nError: {response.status_code}")
        try:
            error_data = response.json()
            print(f"Detalle del error: {error_data.get('detail', 'No detail provided')}")
        except:
            print(f"Response text: {response.text}")
except requests.exceptions.ConnectionError:
    print("\nError de conexión: No se pudo conectar al servidor.")
    print("Asegúrate de que el servidor esté corriendo con 'uvicorn app_new:app --reload'")
except Exception as e:
    print(f"\nError al hacer la petición: {str(e)}")
    import traceback
    print(f"Traceback: {traceback.format_exc()}")
