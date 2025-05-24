import pandas as pd
from fastapi import APIRouter, HTTPException
from pathlib import Path
from typing import List

from model.schemas import Question

router = APIRouter()

# Cargar preguntas
def load_questions() -> List[Question]:
    try:
        # Cargar todas las preguntas
        questions_df = pd.read_csv(Path(__file__).parent.parent.parent / "Preguntas.csv")
        
        # Cargar índices de las preguntas top 80
        with open(Path(__file__).parent.parent.parent / "preguntas_top_80.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
            # Extraer números de pregunta (Q58, Q6, etc)
            top_questions = []
            for line in lines[2:]:  # Skip first 2 lines
                if line.strip() and line[0].isdigit():
                    # Extraer el número después de la Q
                    q_num = int(line.strip().split('. Q')[1])
                    top_questions.append(q_num)
        
        # Filtrar solo las preguntas top 80
        questions = []
        for q_num in top_questions:
            # Buscar la pregunta en el DataFrame
            question_row = questions_df[questions_df['No Pregunta'] == q_num].iloc[0]
            questions.append(
                Question(
                    id=int(question_row["No Pregunta"]),
                    text=question_row["Pregunta"],
                    enneagram_type=int(question_row["eneatipo"])
                )
            )
        
        return questions
    except Exception as e:
        print(f"Error loading questions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/questions", response_model=List[Question])
async def get_questions():
    """Obtener la lista de las 80 preguntas del test"""
    questions = load_questions()
    return questions
