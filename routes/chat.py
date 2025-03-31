from fastapi import APIRouter, Request, Depends, Security, HTTPException
from schemas import ChatCompletionRequest, ImageRequest, ChatMessage
from typing import Optional
from config import config
from auth import verify_auth
from .images import generate_image_internal
from sklearn.linear_model import LogisticRegression
from pymorphy3 import MorphAnalyzer
import requests
import uuid
import time
import logging
import os
import re
import nltk
from nltk.corpus import stopwords
import joblib

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)

nltk.download('stopwords')
russian_stopwords = stopwords.words('russian')
morph = MorphAnalyzer()
model = joblib.load('resources/classifier.joblib')
vectorizer = joblib.load('resources/vectorizer.joblib')

OLLAMA_BASE_URL = f"http://{config.ollama.host}:{config.ollama.port}"
OLLAMA_CHAT_ENDPOINT = f"{OLLAMA_BASE_URL}/api/chat"

@router.post("/chat/completions", dependencies=[Depends(verify_auth)])
async def chat_completion(
        fastapi_request: Request,
        request_body: ChatCompletionRequest
    ):
    try:
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

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Удаление пунктуации
    tokens = text.split()
    
    # Лемматизация и фильтрация стоп-слов
    tokens = [morph.parse(token)[0].normal_form for token in tokens]
    tokens = [token for token in tokens if token not in russian_stopwords]
    
    return ' '.join(tokens)

def is_image_request(request: ChatCompletionRequest) -> bool:
    last_user_message = get_last_user_message(request.messages)
    if not last_user_message:
        return False
    processed_text = preprocess_text(last_user_message.content)
    features = vectorizer.transform([processed_text])
    prediction = model.predict(features)[0]
    return bool(prediction)        

def get_last_user_message(messages: list[ChatMessage]) -> Optional[ChatMessage]:
    for msg in reversed(messages):
        if msg.role == "user":
            return msg
    return None

async def handle_image_generation(
        request: Request,
        chat_request: ChatCompletionRequest
    ) -> dict:
    original_prompt = get_last_user_message(chat_request.messages)
    if not original_prompt:
        raise HTTPException(400, "No user message found")

    generated_prompt = await generate_image_prompt(chat_request, original_prompt)
    
    image_response = await generate_and_save_image(
        prompt=generated_prompt,
        language='auto'
    )
    
    return create_chat_image_response(
        request=request,
        prompt=image_response["prompt"],
        image_path=image_response["filepath"]
    )

async def generate_image_prompt(
        chat_request: ChatCompletionRequest,
        original_prompt: str
    ) -> str:
    base_system_message = ""
    for msg in chat_request.messages:
        if (msg.role == "system"):
            base_system_message = base_system_message + msg.content + "\n"
        
    system_message = ChatMessage(
        role="system",
        content=base_system_message + "Сейчас ты помогаешь создавать промпты для генерации изображений."
    )
    user_message = ChatMessage(
        role="user",
        content=f"Составь промпт для генерации изображения размером до 150 слов, основываясь на данном запросе: '{original_prompt}'"
    )
    
    payload = {
        "model": chat_request.model,
        "messages": [m.dict() for m in [system_message, user_message]],
        "temperature": chat_request.temperature if chat_request.temperature is not None else 0.7,
        "top_p": chat_request.top_p if chat_request.top_p is not None else 1.0,
        "max_tokens": chat_request.max_tokens,
        "stream": False
    }
    
    try:
        ollama_response = await send_ollama_request(payload)
        return ollama_response['message']['content'].strip()
    except Exception as e:
        logger.error(f"Prompt generation error: {e}")
        return original_prompt

async def generate_and_save_image(prompt: str, language: str) -> dict:
    image_request = ImageRequest(
        prompt=prompt,
        model='black-forest-labs/FLUX.1-dev',
        steps=50,
        size='512x512',
        guidance_scale = 7.5
    )
    return await generate_image_internal(image_request, language)

def create_chat_image_response(request: Request, prompt: str, image_path: str) -> dict:
    filename = os.path.basename(image_path)
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "image-generator",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": f"![image]({request.base_url}v1/images/{filename}) {prompt}"
            }
        }],
        "usage": create_image_response_approx_usage(prompt)
    }

async def handle_text_completion(request_body: ChatCompletionRequest) -> dict:
    ollama_payload = prepare_ollama_payload(request_body)
    ollama_response = await send_ollama_request(ollama_payload)
    return format_openai_response(request_body, ollama_response)

def prepare_ollama_payload(request: ChatCompletionRequest) -> dict:
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
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [format_choice(ollama_response)],
        "usage": calculate_usage(ollama_response)
    }

def format_choice(ollama_response: dict) -> dict:
    return {
        "index": 0,
        "message": {
            "role": ollama_response["message"]["role"],
            "content": ollama_response["message"]["content"]
        },
        "finish_reason": "stop" if ollama_response["done"] else "length"
    }

def calculate_usage(ollama_response: dict) -> dict:
    return {
        "prompt_tokens": ollama_response.get("prompt_eval_count", 0),
        "completion_tokens": ollama_response.get("eval_count", 0),
        "total_tokens": (
            ollama_response.get("prompt_eval_count", 0) +
            ollama_response.get("eval_count", 0)
        )
    }

def create_image_response_approx_usage(prompt: str) -> dict:
    wordCount = len(prompt.split())
    return {
        "prompt_tokens": wordCount,
        "completion_tokens": 0,
        "total_tokens": wordCount
    }

def log_error(error: Exception):
    logger.error(f"Error processing request: {str(error)}", exc_info=True)