from functools import lru_cache
from src.services.model_service import ModelService

@lru_cache()
def get_model_service() -> ModelService:
    """
    Dependency that provides a cached instance of ModelService.
    The lru_cache decorator ensures we only create one instance.
    """
    return ModelService()
