from typing import Set, Dict, List
from fastapi import HTTPException
import logging
from services.models_service import get_all_models
import asyncio

logger = logging.getLogger(__name__)

class ModelCache:
    def __init__(self):
        self._cache: Set[str] = set()
        self._lock = asyncio.Lock()
    
    async def refresh(self):
        async with self._lock:
            try:
                models = await asyncio.to_thread(get_all_models)
                self._cache = {model["id"] for model in models}
                logger.info(f"Model cache updated. Current models: {len(self._cache)}")
            except Exception as e:
                logger.error(f"Error refreshing model cache: {str(e)}", exc_info=True)
    
    def contains(self, model_id: str) -> bool:
        return model_id in self._cache
    
    def validate_model(self, model_id: str):
        if not self.contains(model_id):
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_id}' not found in local cache. Please download first."
            )

# Инициализация глобального кеша
model_cache = ModelCache()