from schemas.chat import ChatCompletionRequest
from config import config
from .ollama_client import OllamaClient
from typing import Optional
from fastapi import HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import logging
import uuid
import time
import gc
import re

logger = logging.getLogger(__name__)

class ChatTextService:
    def __init__(self):
        self.ollama = OllamaClient()

    async def process_text_request(self, body: ChatCompletionRequest, provider: str):
        if provider == "ollama":
            return await self._process_ollama_request(body)
        elif provider == "huggingface":
            return await self._process_huggingface_request(body)
        raise HTTPException(400, "Unsupported model provider")

    async def _process_ollama_request(self, body: ChatCompletionRequest):
        logger.info(f"Passing chat request to Ollama backend. Language model applied: {body.model}")
        payload = self._prepare_ollama_payload(body)
        response = await self.ollama.chat(payload)
        return self._format_response(body, response)

    async def _process_huggingface_request(self, body: ChatCompletionRequest):
        
        model_id = body.model
        messages = [m.dict() for m in body.messages]
        
        try:
            logger.info(f"Loading Hugging Face model and tokenizer: {model_id}")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                offload_state_dict=True
            )
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer
            )

            logger.info(f"Passing chat request to Hugging Face model: {model_id}")
            prompt = self._format_messages(messages)
            generator = pipe(
                prompt,
                max_new_tokens=body.max_tokens,
                temperature=body.temperature,
                top_p=body.top_p,
                return_full_text=False
            )
            content = generator[0]['generated_text']
            
            del pipe
            del generator
            torch.cuda.empty_cache()

            return self._format_response(body, {
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "done": True
            })
            
        except Exception as e:
            logger.error(f"Hugging Face inference error: {str(e)}")
            raise HTTPException(500, "Model processing failed")
        
        finally:
            # Гарантированная очистка ресурсов
            self._cleanup_hf_artifacts(model, tokenizer)

    def _format_messages(self, messages: list) -> str:
        return "\n".join(
            [f"{msg['role'].upper()}: {msg['content']}" 
             for msg in messages]
        ) + "\nASSISTANT: "

    def _prepare_ollama_payload(self, request: ChatCompletionRequest):
        messages = []
        for msg in request.messages:
            if isinstance(msg.content, list):
                for c in msg.content:
                    if c.type == "text":
                        messages.append({"role": msg.role, "content": c.text})
            else:
                messages.append(msg.dict())
        return {
            "model": request.model,
            "messages": messages,
            "options": {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "num_predict": request.max_tokens
            },
            "stream": request.stream
        }

    def _format_response(self, request: ChatCompletionRequest, response: dict):
        logger.info("Processing text model response")

        # Извлекаем оригинальное сообщение
        message = response["message"].copy()
        original_content = message["content"]
        
        # Обрабатываем секции с размышлениями
        processed_content = self._process_reflections(original_content)
        
        # Создаем финальное сообщение
        message["content"] = processed_content
        
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": "stop" if response["done"] else "length"
            }],
            "usage": self._calculate_usage(response)
        }

    def _process_reflections(self, content: str) -> str:
        """Форматирует секции размышлений в свернутые блоки"""
        # Регулярное выражение для поиска секций
        pattern = re.compile(
            r'<think>(.*?)</think>',
            re.DOTALL  # Для захвата многострочных блоков
        )
        
        # Замена на HTML-теги
        processed = pattern.sub(
            self._wrap_reflection,
            content
        )
        
        return processed.strip()

    def _wrap_reflection(self, match: re.Match) -> str:
        """Оборачивает размышления в теги details/summary"""
        reflection_text = match.group(1).strip()
        return (
            '\n\n<details class="reflection">'
            '<summary>Размышления модели</summary>'
            f'{reflection_text}'
            '</details>'
        )

    def _calculate_usage(self, response: dict):
        return {
            "prompt_tokens": response.get("prompt_eval_count", 0),
            "completion_tokens": response.get("eval_count", 0),
            "total_tokens": (
                response.get("prompt_eval_count", 0) +
                response.get("eval_count", 0)
            )
        }

    def _cleanup_hf_artifacts(self, model, tokenizer):
        try:
            # Очистка модели
            model.to('cpu')
            del model
            # Очистка токенизатора
            del tokenizer
            # Принудительный сбор мусора
            gc.collect()
            torch.cuda.empty_cache()
            # Дополнительная очистка памяти
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
        except Exception as e:
            logger.warning(f"Cleanup error: {str(e)}")