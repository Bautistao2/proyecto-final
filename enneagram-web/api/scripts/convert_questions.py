import csv
import json
from pathlib import Path
import re

def convert_csv_to_json():
    # Definir rutas absolutas
    base_path = Path("C:/Users/bauti/Documents/claude/enneagram-web")
    top_80_file = base_path / "preguntas_top_80.txt"
    preguntas_file = base_path / "Preguntas.csv"
    output_file = base_path / "api/frontend/static/data/questions.json"

    print(f"Leyendo preguntas top 80 desde: {top_80_file}")
    print(f"Leyendo CSV desde: {preguntas_file}")
    print(f"Guardando JSON en: {output_file}")

    # Asegurarse que el directorio existe
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Leer las preguntas top 80
    top_80_questions = []
    pattern = r'Q(\d+)'  # Patrón para encontrar "Q" seguido de números
    
    with open(top_80_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                num = int(match.group(1))  # Obtener solo el número
                top_80_questions.append(num)
                print(f"Encontrada pregunta Q{num}")  # Debug

    print(f"\nEncontradas {len(top_80_questions)} preguntas top 80")
    print(f"Lista de preguntas encontradas: {top_80_questions}")

    # Leer el CSV y convertir a JSON
    questions = []
    question_map = {}
    
    with open(preguntas_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            num = int(row['No Pregunta'])
            question_map[num] = {
                'id': num,
                'text': row['Pregunta'],
                'eneatipo': int(row['eneatipo'])
            }
    
    # Construir la lista final de preguntas en el orden correcto
    for num in top_80_questions:
        if num in question_map:
            questions.append(question_map[num])
            print(f"Agregada pregunta {num}: {question_map[num]['text'][:50]}...")
        else:
            print(f"¡ADVERTENCIA! No se encontró la pregunta {num} en el CSV")

    print(f"\nProcesadas {len(questions)} preguntas")

    # Guardar como JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"Archivo JSON creado exitosamente en {output_file}")
    print(f"Total de preguntas en el archivo JSON: {len(questions)}")

if __name__ == '__main__':
    convert_csv_to_json()
