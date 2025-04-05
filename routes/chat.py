from fastapi import APIRouter, Request, Depends, Security, HTTPException
from schemas import ChatCompletionRequest
from services.chat_service import ChatService
from auth import verify_auth
from models_cache import model_cache  # <-- Добавляем импорт кеша
import logging

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)
chat_service = ChatService()

@router.post("/chat/completions", dependencies=[Depends(verify_auth)])
async def chat_completion(
        request: Request,
        body: ChatCompletionRequest
    ):
    try:
        model_cache.validate_model(body.model)
        provider = model_cache.get_provider(body.model)
        return await chat_service.process_chat_request(
            body, 
            provider=provider
        )
    except HTTPException as he:
        # Пробрасываем HTTP исключения напрямую
        raise he
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(500, detail=str(e))