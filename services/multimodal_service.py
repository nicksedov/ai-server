from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
import base64
import io
import logging
import uuid
import time
import gc
from contextlib import contextmanager
from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)

class GPUMemoryManager:
    @staticmethod
    @contextmanager
    def scope():
        try:
            yield
        finally:
            torch.cuda.empty_cache()
            gc.collect()
            logger.debug("GPU memory cleared after scope")

class MultimodalService:
    def __init__(self):
        self.ollama_client = OllamaClient()
        self._memory_logging = True

    async def process_request(self, request, provider):
        if provider == "huggingface":
            with GPUMemoryManager.scope():
                return await self._process_huggingface_request(request)
        elif provider == "ollama":
            return await self._process_ollama_request(request)
        raise HTTPException(400, "Unsupported provider for multimodal model")

    # Hugging Face model processing
    async def _process_huggingface_request(self, request):
        model_id = request.model
        self.log_memory_usage("Before model loading")
        
        processor = self._load_processor(model_id)
        model = self._load_model(model_id)
        self.log_memory_usage("After model loading")

        try:
            inputs = await self._prepare_inputs(processor, request.messages)
            self.log_memory_usage("After inputs preparation")

            with torch.no_grad(), self._timed_block("Model inference"):
                generated_ids = model.generate(**inputs, max_new_tokens=512)
            
            generated_text = processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]

            self._cleanup_tensors(inputs, generated_ids)
            return self._format_response(request, generated_text)
        finally:
            self._cleanup_resources(model, processor)
            self.log_memory_usage("After cleanup")

    def _load_model(self, model_id):
        logger.info(f"Loading multimodal model: {model_id}")
        return AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        )

    def _load_processor(self, model_id):
        logger.info(f"Loading model processor: {model_id}")
        return AutoProcessor.from_pretrained(model_id)

    async def _prepare_inputs(self, processor, messages):
        text = []
        images = []
        
        for msg in messages:
            if isinstance(msg.content, list):
                for item in msg.content:
                    if item.type == "text":
                        text.append(item.text)
                    elif item.type == "image_url":
                        images.append(await self._decode_image(item.image_url["url"]))
            else:
                text.append(msg.content)
        
        # Вызываем новую функцию для формирования промпта
        final_prompt = self._build_final_prompt(
            text_parts=text,
            num_images=len(images)
        )
        inputs = processor(
            text=final_prompt,
            images=images if images else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to("cuda")

        return inputs

    def _build_final_prompt(self, text_parts: list, num_images: int) -> str:
        """Формирует финальный промпт с правильным количеством тегов <image>"""
        # Извлекаем последний текстовый элемент или используем дефолтный промпт
        original_prompt = text_parts[-1] if text_parts else "Describe the image"
        
        # Очищаем от существующих тегов
        cleaned_prompt = original_prompt.replace("<image>", "").strip()
        
        # Генерируем необходимое количество тегов
        image_tags = ' '.join(['<image>'] * num_images) if num_images > 0 else ''
        
        # Комбинируем части
        if cleaned_prompt and image_tags:
            return f"{cleaned_prompt} {image_tags}".strip()
        if image_tags:
            return image_tags
        return cleaned_prompt or "Describe the image"
        
    def _cleanup_tensors(self, inputs, generated_ids):
        """Исправленная версия с безопасной обработкой словаря"""
        # Создаем копию ключей для безопасной итерации
        keys = list(inputs.keys())
        
        for key in keys:
            tensor = inputs[key]
            if isinstance(tensor, torch.Tensor):
                # Переносим тензор на CPU и удаляем
                inputs[key] = tensor.detach().cpu()
                del tensor  # Явное удаление ссылки
                del inputs[key]  # Удаляем ключ из словаря после обработки
        
        # Обрабатываем generated_ids отдельно
        if generated_ids is not None:
            generated_ids = generated_ids.detach().cpu()
            del generated_ids

        # Принудительная очистка памяти
        torch.cuda.empty_cache()
        
    def _cleanup_resources(self, model, processor):
        """Дополненная очистка ресурсов"""
        # Удаление модели
        if model is not None:
            if hasattr(model, "cpu"):
                model.cpu()
            del model
        
        # Удаление процессора
        if processor is not None:
            del processor
        
        # Дополнительные меры очистки
        time.sleep(0.5)
        self.flush()

    def flush(self):
        """Comprehensive memory cleanup"""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
                torch.cuda.reset_accumulated_memory_stats(i)
            torch.cuda.empty_cache()
        gc.collect()

    def log_memory_usage(self, prefix=""):
        if self._memory_logging and torch.cuda.is_available():
            logger.info(
                f"{prefix} - "
                f"Allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB, "
                f"Reserved: {torch.cuda.memory_reserved()/1024**2:.2f}MB, "
                f"Max Allocated: {torch.cuda.max_memory_allocated()/1024**2:.2f}MB"
            )

    @contextmanager
    def _timed_block(self, name):
        start = time.time()
        try:
            yield
        finally:
            logger.info(f"{name} took {time.time() - start:.2f}s")

    async def _decode_image(self, url):
        if url.startswith("data:image"):
            image_data = url.split(",")[1]
            image = Image.open(io.BytesIO(base64.b64decode(image_data)))
            return image.convert("RGB")
        # Implement other image sources here

    def _format_response(self, request, text):
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop"
            }]
        }

    # Ollama model processing
    async def _process_ollama_request(self, request):
        messages = await self._prepare_ollama_messages(request.messages)
        payload = {
            "model": request.model,
            "messages": messages,
            "stream": request.stream,
            "options": {
                "temperature": request.temperature,
                "top_p": request.top_p
            }
        }
        
        try:
            response = await self.ollama_client.chat(payload)
            return self._format_response(request, response['message']['content'])
        except Exception as e:
            logger.error(f"Ollama processing failed: {str(e)}")
            raise HTTPException(500, "Multimodal processing error")

    async def _prepare_ollama_messages(self, messages):
        prepared = []
        for msg in messages:
            if isinstance(msg.content, list):
                content = []
                images = []
                for item in msg.content:
                    if item.type == "text":
                        content.append(item.text)
                    elif item.type == "image_url":
                        base64_image = await self._decode_image_to_base64(item.image_url['url'])
                        images.append(base64_image)
                prepared.append({
                    "role": msg.role,
                    "content": "\n".join(content),
                    "images": images
                })
            else:
                prepared.append(msg.dict())
        return prepared

    async def _decode_image_to_base64(self, url: str) -> str:
        if url.startswith("data:image"):
            return url.split(",")[1]
        raise HTTPException(400, "Only embedded images are supported")