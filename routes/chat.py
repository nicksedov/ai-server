from fastapi import APIRouter, Request, Depends, Security, HTTPException
from schemas.chat import ChatCompletionRequest, ChatMessage
from services.classifier_service import PromptClassifier
from services.chat_text_service import ChatTextService
from services.chat_image_service import ChatImageService
from auth import verify_auth
from models_cache import model_cache
import logging

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)
classifier = PromptClassifier()
chat_service = ChatTextService()
image_chat_service = ChatImageService()

@router.post("/chat/completions", 
            dependencies=[Depends(verify_auth)],
            summary="Генерация текстового ответа или изображения",
            description="""Основной эндпоинт для взаимодействия с языковыми моделями. Поддерживает:
- Текстовые ответы через модели Ollama
- Автоматическое преобразование текстовых запросов в изображения
- Поддержка стриминга через параметр stream""")
async def chat_completion(
        request: Request,
        body: ChatCompletionRequest
    ):
    try:
        model_cache.validate_model(body.model)
        provider = model_cache.get_provider(body.model)
        if is_image_request(body):
            return await image_chat_service.handle_image_request(body)
        return await chat_service.process_text_request(body)
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(500, detail=str(e))

def is_image_request(request: ChatCompletionRequest) -> bool:
    for msg in reversed(request.messages):
        if msg.role == "user" and msg.content:
            return classifier.is_image_request(msg.content)
    raise HTTPException(400, "No user message found")