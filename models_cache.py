from typing import Set, Dict, List
from fastapi import HTTPException
import logging
from services.model_service import ModelService
import asyncio

logger = logging.getLogger(__name__)
model_service = ModelService()

class ModelCache:
    def __init__(self):
        self._cache: Dict[str, str] = {}  # model_id -> provider
        self._lock = asyncio.Lock()
    
    async def refresh(self):
        async with self._lock:
            try:
                models = await model_service.get_all_models()
                self._cache = {m["id"]: m["owned_by"] for m in models}
                logger.info(f"Model cache updated. Models: {len(self._cache)}")
            except Exception as e:
                logger.error(f"Error refreshing model cache: {str(e)}")

    def get_provider(self, model_id: str) -> str:
        return self._cache.get(model_id)
    
    def validate_model(self, model_id: str):
        if model_id not in self._cache:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_id}' not found"
            )
            
# Инициализация глобального кеша
model_cache = ModelCache()