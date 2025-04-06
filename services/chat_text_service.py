from schemas.chat import ChatCompletionRequest
from config import config
from .ollama_client import OllamaClient
from fastapi import HTTPException
import logging
import uuid
import time

logger = logging.getLogger(__name__)

class ChatTextService:
    def __init__(self):
        self.ollama = OllamaClient()

    async def process_text_request(self, request_body: ChatCompletionRequest):
        payload = self.prepare_ollama_payload(request_body)
        ollama_response = await self.ollama.chat(payload)
        return self.format_openai_response(request_body, ollama_response)

    def prepare_ollama_payload(self, request: ChatCompletionRequest):
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

    def format_openai_response(self, request: ChatCompletionRequest, ollama_response: dict):
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [self.format_choice(ollama_response)],
            "usage": self.calculate_usage(ollama_response)
        }

    @staticmethod
    def format_choice(response: dict):
        return {
            "index": 0,
            "message": {
                "role": response["message"]["role"],
                "content": response["message"]["content"]
            },
            "finish_reason": "stop" if response["done"] else "length"
        }

    @staticmethod
    def calculate_usage(response: dict):
        return {
            "prompt_tokens": response.get("prompt_eval_count", 0),
            "completion_tokens": response.get("eval_count", 0),
            "total_tokens": (
                response.get("prompt_eval_count", 0) +
                response.get("eval_count", 0)
            )
        }