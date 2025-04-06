from schemas import ChatCompletionRequest, ChatMessage
from config import config
from .ollama_client import OllamaClient
from .image_service import ImageService
from .classifier_service import PromptClassifier
from fastapi import HTTPException
import logging
import uuid
import time
import os

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self):
        self.ollama = OllamaClient()
        self.image_service = ImageService()
        self.classifier = PromptClassifier()

    async def process_chat_request(self, request_body: ChatCompletionRequest, provider: str):
        if (provider != 'ollama'):
            raise HTTPException(400, f"Model type '{provider}' not supported")
        if self.is_image_request(request_body):
            return await self.handle_image_generation(request_body)
        return await self.handle_text_completion(request_body)

    def is_image_request(self, request: ChatCompletionRequest) -> bool:
        last_user_message = self.get_last_user_message(request.messages)
        if not last_user_message:
            return False
        return self.classifier.is_image_request(last_user_message.content)

    @staticmethod
    def preprocess_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        tokens = [morph.parse(token)[0].normal_form for token in tokens]
        return ' '.join([t for t in tokens if t not in russian_stopwords])

    @staticmethod
    def get_last_user_message(messages: list[ChatMessage]):
        for msg in reversed(messages):
            if msg.role == "user":
                return msg
        return None

    async def handle_image_generation(self, chat_request: ChatCompletionRequest):
        original_prompt = self.get_last_user_message(chat_request.messages)
        if not original_prompt:
            raise HTTPException(400, "No user message found")

        generated_prompt = await self.generate_image_prompt(chat_request, original_prompt)
        time.sleep(1) # Take time for Ollama to unload its background process
        image_response = await self.image_service.generate_and_save_image(
            model="black-forest-labs/FLUX.1-dev",
            prompt=generated_prompt,
            language='auto',
            size="512x512",
            steps=50,
            guidance_scale=4.5           
        )
        return self.create_chat_image_response(
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

    async def handle_text_completion(self, request_body: ChatCompletionRequest):
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

    def create_chat_image_response(self, prompt: str, image_path: str):
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
            "usage": self.create_image_response_approx_usage(prompt)
        }

    @staticmethod
    def create_image_response_approx_usage(prompt: str):
        word_count = len(prompt.split())
        return {
            "prompt_tokens": word_count,
            "completion_tokens": 0,
            "total_tokens": word_count
        }