from fastapi import APIRouter, Request, Depends, Security, HTTPException
from schemas.chat import ChatCompletionRequest, ChatMessage
from services.bert_prompt_classifier import BertPromptClassifier
from services.chat_text_service import ChatTextService
from services.chat_image_service import ChatImageService
from services.multimodal_service import MultimodalService
from auth import verify_auth
from models_cache import model_cache
import logging

router = APIRouter(prefix="/v1", tags=['chat'])
logger = logging.getLogger(__name__)
classifier = BertPromptClassifier()
chat_service = ChatTextService()
image_chat_service = ChatImageService()
multimodal_service = MultimodalService()

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
           
        if model_cache.is_multimodal(body.model) and is_image_attached(body):
            return await multimodal_service.process_request(body, provider)

        if is_image_request(body):
            return await image_chat_service.handle_image_request(body)

        return await chat_service.process_text_request(body, provider)
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(500, detail=str(e))

def is_image_attached(request: ChatCompletionRequest) -> bool:
    for msg in reversed(request.messages):
        if msg.role == "user" and msg.content:
            if isinstance(msg.content, list):
                for item in msg.content:
                    if item.type == "image_url" and item.image_url:
                        return True
            return False


def is_image_request(request: ChatCompletionRequest) -> bool:
    text = ''
    for msg in reversed(request.messages):
        if msg.role == "user" and msg.content:
            if isinstance(msg.content, list):
                for item in msg.content:
                    if item.type == "text":
                        text = '\n'.join([text, item.text])
            elif isinstance(msg.content, str):
                text = msg.content
            return classifier.is_image_request(text)
    raise HTTPException(400, "No user message found")