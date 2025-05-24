from fastapi import APIRouter, HTTPException, status
from src.schemas.question import QuestionsResponse, Question
from src.data.questions import QUESTIONS as RAW_QUESTIONS

router = APIRouter()

# Convertir las preguntas del formato raw al formato de Pydantic
QUESTIONS = [
    Question(
        id=q["id"], 
        text=q["text"], 
        category=q["category"],
        original_id=q["original_id"]
    ) for q in RAW_QUESTIONS
]

@router.get("/questions", 
    response_model=QuestionsResponse,
    summary="Obtener preguntas del test",
    description="Devuelve la lista de preguntas del test de eneagrama"
)
async def get_questions():
    """
    Retorna la lista completa de preguntas del test de eneagrama.
    """
    try:
        if not QUESTIONS or len(QUESTIONS) == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No se encontraron preguntas disponibles.")
        return QuestionsResponse(
            questions=QUESTIONS,
            total=len(QUESTIONS)
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error al obtener preguntas: {str(e)}")
