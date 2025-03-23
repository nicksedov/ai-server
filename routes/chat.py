from fastapi import APIRouter, Request, Depends, Security, HTTPException
from schemas import ChatCompletionRequest, ImageRequest, ChatMessage
from typing import Optional
from config import config
from auth import verify_auth
from .images import generate_image_internal
import requests
import uuid
import time
import logging
import os

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = f"http://{config.ollama.host}:{config.ollama.port}"
OLLAMA_CHAT_ENDPOINT = f"{OLLAMA_BASE_URL}/api/chat"

# Chat completions API
#
# POST /v1/chat/completions
# Headers:
#   Content-Type: application/json
#   Authorization: Bearer <your_api_key>
# Body:
#   {
#     "model": "<ollama-model-name>",
#     "messages": [
#       {"role": "system|user|assistant", "content": "<message-text>"},
#       ...
#     ],
#     "temperature": <0.0-1.0>,    # Optional (default: 0.7)
#     "top_p": <0.0-1.0>,          # Optional (default: 1.0)
#     "max_tokens": <n>,           # Optional (default: unlimited)
#     "stream": <bool>             # Optional (default: false)
#   }
#
# Example of API call
# curl -X POST -H 'Content-Type: application/json' \
#  -d '{
#    "model": "llama2",
#    "messages": [
#      {"role": "user", "content": "Tell me a joke about AI"}
#    ],
#    "temperature": 0.7
#  }' \
#  http://localhost:8000/v1/chat/completions
#
# Note: Requires running Ollama server with specified model (e.g. `ollama pull llama2`)
#
@router.post("/chat/completions", dependencies=[Depends(verify_auth)])
async def chat_completion(
        fastapi_request: Request,
        request_body: ChatCompletionRequest
    ):
    try:
        # Основная логика маршрутизации
        if is_image_request(request_body):
            return await handle_image_generation(fastapi_request, request_body)
        else:
            return await handle_text_completion(request_body)
            
    except HTTPException:
        raise
    except Exception as e:
        log_error(e)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# ======================
# Вспомогательные функции
# ======================

def is_image_request(request: ChatCompletionRequest) -> bool:
    """Определяет содержит ли запрос требование генерации изображения"""
    last_user_message = get_last_user_message(request.messages)
    if not last_user_message:
        return False
    
    keywords = ["нарисуй", "изобрази", "сгенерируй изображение", "нарисуйте", "покажи изображение"]
    return any(keyword in last_user_message.content.lower() for keyword in keywords)

def get_last_user_message(messages: list[ChatMessage]) -> Optional[ChatMessage]:
    """Возвращает последнее сообщение пользователя"""
    for msg in reversed(messages):
        if msg.role == "user":
            return msg
    return None

async def handle_image_generation(
        request: Request,
        chat_request: ChatCompletionRequest
    ) -> dict:
    """Обрабатывает запросы на генерацию изображений"""
    image_prompt = extract_image_prompt(chat_request)
    content_language = detect_content_language(image_prompt)
    
    image_response = await generate_and_save_image(
        prompt=image_prompt,
        language=content_language
    )
    
    return create_chat_image_response(
        request=request,
        prompt=image_prompt,
        image_path=image_response["filepath"]
    )

def extract_image_prompt(chat_request: ChatCompletionRequest) -> str:
    """Извлекает промпт для генерации изображения из запроса"""
    last_user_message = get_last_user_message(chat_request.messages)
    if not last_user_message:
        raise HTTPException(400, "No user message found")
    
    # Удаляем ключевые слова из промпта
    keywords = ["нарисуй", "изобрази", "сгенерируй изображение", "нарисуйте", "покажи изображение"]
    cleaned_prompt = last_user_message.content
    for kw in keywords:
        cleaned_prompt = cleaned_prompt.replace(kw, "").replace(kw.capitalize(), "")
    
    return cleaned_prompt.strip()

def detect_content_language(prompt: str) -> str:
    """Определяет язык введенного промпта"""
    try:
        return Translator().detect(prompt).lang or 'en'
    except:
        return 'en'

async def generate_and_save_image(prompt: str, language: str) -> dict:
    """Генерация и сохранение изображения"""
    image_request = ImageRequest(
        prompt=prompt,
        model='black-forest-labs/FLUX.1-dev',
        steps=50,
        size='512x512'
    )
    return await generate_image_internal(image_request, language)

def create_chat_image_response(request: Request, prompt: str, image_path: str) -> dict:
    """Формирует ответ чата с изображением"""
    filename = os.path.basename(image_path)
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "image-generator",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": f"Изображение сгенерировано: {filename}",
                "image_url": f"{request.base_url}v1/images/{filename}"
            }
        }],
        "usage": create_usage_stats(prompt)
    }

async def handle_text_completion(request_body: ChatCompletionRequest) -> dict:
    """Обрабатывает текстовые запросы"""
    ollama_payload = prepare_ollama_payload(request_body)
    ollama_response = await send_ollama_request(ollama_payload)
    return format_openai_response(request_body, ollama_response)

def prepare_ollama_payload(request: ChatCompletionRequest) -> dict:
    """Подготавливает payload для запроса к Ollama"""
    return {
        "model": request.model,
        "messages": [m.dict() for m in request.messages],
        "options": {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "num_predict": request.max_tokens
        },
        "stream": request.stream
    }

async def send_ollama_request(payload: dict) -> dict:
    """Отправляет запрос к Ollama API"""
    try:
        response = requests.post(
            OLLAMA_CHAT_ENDPOINT,
            json=payload,
            timeout=config.ollama.timeout
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ReadTimeout:
        raise HTTPException(504, "Ollama response timeout")
    except requests.exceptions.RequestException as e:
        raise HTTPException(500, f"Ollama API error: {str(e)}")

def format_openai_response(
        request: ChatCompletionRequest,
        ollama_response: dict
    ) -> dict:
    """Форматирует ответ Ollama в OpenAI-совместимый формат"""
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [format_choice(ollama_response)],
        "usage": calculate_usage(ollama_response)
    }

def format_choice(ollama_response: dict) -> dict:
    """Форматирует выбор ответа"""
    return {
        "index": 0,
        "message": {
            "role": ollama_response["message"]["role"],
            "content": ollama_response["message"]["content"]
        },
        "finish_reason": "stop" if ollama_response["done"] else "length"
    }

def calculate_usage(ollama_response: dict) -> dict:
    """Вычисляет статистику использования токенов"""
    return {
        "prompt_tokens": ollama_response.get("prompt_eval_count", 0),
        "completion_tokens": ollama_response.get("eval_count", 0),
        "total_tokens": (
            ollama_response.get("prompt_eval_count", 0) +
            ollama_response.get("eval_count", 0)
        )
    }

def create_usage_stats(prompt: str) -> dict:
    """Генерирует статистику использования для изображений"""
    return {
        "prompt_tokens": len(prompt.split()),
        "completion_tokens": 0,
        "total_tokens": len(prompt.split())
    }

def log_error(error: Exception):
    """Логирует ошибки"""
    logger.error(f"Error processing request: {str(error)}", exc_info=True)