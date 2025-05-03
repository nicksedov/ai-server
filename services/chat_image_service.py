from schemas.chat import ChatCompletionRequest, ChatMessage
from config import config
from .ollama_client import OllamaClient
from .image_service import ImageService
from fastapi import HTTPException
import logging
import uuid
import time
import os

logger = logging.getLogger(__name__)

class ChatImageService:
    def __init__(self):
        self.ollama = OllamaClient()
        self.image_service = ImageService()

    async def handle_image_request(self, chat_request: ChatCompletionRequest):
        for msg in reversed(chat_request.messages):
            if msg.role == "user" and msg.content:
                system_message = self.build_system_message(chat_request.messages)
                generated_prompt = await self.generate_image_prompt(
                    system_message, 
                    msg.content
                )
                image_response = await self.image_service.generate_and_save_image(
                    model=config.chat.image_generation.default_model,
                    prompt=generated_prompt,
                    language=config.chat.image_generation.default_language,
                    size=config.chat.image_generation.default_size,
                    steps=config.chat.image_generation.default_steps,
                    guidance_scale=config.chat.image_generation.default_guidance_scale
                )
                return self.create_image_response(
                    prompt=image_response["prompt"],
                    image_path=image_response["filepath"]
                )

    async def generate_image_prompt(self, system_prompt: str, original_prompt: str):
        system_message = ChatMessage(
            role="system", 
            content=system_prompt
        )
        user_message = ChatMessage(
            role="user",
            content=f"Создай промпт для генерации изображения размером до 150 слов, основываясь на данном запросе: '{original_prompt}'"
        )
        
        payload = {
            "model": config.ollama.default_model,
            "messages": [m.dict() for m in [system_message, user_message]],
            "temperature": config.chat.image_prompt.temperature,
            "top_p": config.chat.image_prompt.top_p,
            "max_tokens": config.chat.image_prompt.max_tokens,
            "stream": False
        }
        
        try:
            response = await self.ollama.chat(payload)
            return response['message']['content'].strip()
        except Exception as e:
            logger.error(f"Prompt generation error: {e}")
            return original_prompt

    def build_system_message(self, messages: list[ChatMessage]):
        system_messages = "\n".join(
            [msg.content for msg in messages if msg.role == "system"]
        )
        return f"{system_messages}\n{config.chat.image_prompt.system_message}"

    def create_image_response(self, prompt: str, image_path: str):
        filename = os.path.basename(image_path)
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "image-generator",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": f"![image](/v1/images/{filename})\n\n<details><summary>Описание</summary>{prompt}</details>"
                }
            }],
            "usage": self._approx_usage(prompt)
        }

    @staticmethod
    def _approx_usage(prompt: str):
        word_count = len(prompt.split())
        return {
            "prompt_tokens": word_count,
            "completion_tokens": 0,
            "total_tokens": word_count
        }

