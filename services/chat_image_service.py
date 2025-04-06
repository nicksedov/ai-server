from schemas import ChatCompletionRequest, ChatMessage
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
                generated_prompt = await self.generate_image_prompt(chat_request, msg.content)
                time.sleep(1)
                image_response = await self.image_service.generate_and_save_image(
                    model="black-forest-labs/FLUX.1-dev",
                    prompt=generated_prompt,
                    language='auto',
                    size="512x512",
                    steps=50,
                    guidance_scale=4.5           
                )
                return self.create_image_response(
                    prompt=image_response["prompt"],
                    image_path=image_response["filepath"]
                )

    async def generate_image_prompt(self, chat_request: ChatCompletionRequest, original_prompt: str):
        system_message = self.build_system_message(chat_request.messages)
        user_message = ChatMessage(
            role="user",
            content=f"Создай промпт для генерации изображения размером до 150 слов, основываясь на данном запросе: '{original_prompt}'"
        )
        
        payload = {
            "model": chat_request.model,
            "messages": [m.dict() for m in [system_message, user_message]],
            "temperature": chat_request.temperature or 0.7,
            "top_p": chat_request.top_p or 1.0,
            "max_tokens": chat_request.max_tokens,
            "stream": False
        }
        
        try:
            response = await self.ollama.chat(payload)
            return response['message']['content'].strip()
        except Exception as e:
            logger.error(f"Prompt generation error: {e}")
            return original_prompt

    def build_system_message(self, messages: list[ChatMessage]):
        system_content = "\n".join(
            [msg.content for msg in messages if msg.role == "system"]
        ) + "\nСейчас ты помогаешь создавать промпты для генерации изображений."
        return ChatMessage(role="system", content=system_content)

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
                    "content": f"![image]({filename}) {prompt}"
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

