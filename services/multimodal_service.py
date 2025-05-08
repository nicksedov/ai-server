from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
import base64
import io
import logging
import uuid
import time
import gc

logger = logging.getLogger(__name__)

class MultimodalService:

    async def process_request(self, request, provider):
        model_id = request.model
        processor = self._load_processor(model_id)
        model = self._load_model(model_id)
        
        try:
            # Process messages and generate response
            with torch.no_grad():
                inputs = await self._prepare_inputs(processor, request.messages)
                generated_ids = model.generate(**inputs, max_new_tokens=512)
            
                generated_text = processor.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True
                )[0]

                # Явно освобождаем память
                inputs = {k: v.to("cpu") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                generated_ids = generated_ids.to("cpu")
                
                return self._format_response(request, generated_text)
        finally:
            # Гарантированное освобождение ресурсов
            if 'inputs' in locals():
                del inputs
            if 'generated_ids' in locals():
                del generated_ids
            
            # Удаление модели и процессора
            if 'model' in locals():
                del model
                torch.cuda.empty_cache()
            if 'processor' in locals():
                del processor
            
            self.flush()

    def _load_model(self, model_id):
        logger.info(f"Loading multimodal model: {model_id}")
        return AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
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
        
        final_prompt = text[-1] if text else "You are an assistant who helps describe what is shown in the picture"
        inputs = processor(
            text=final_prompt,
            images=images if images else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to("cuda")

        return inputs

    async def _decode_image(self, url):
        # Extract base64 data
        if url.startswith("data:image"):
            image_data = url.split(",")[1]
            image = Image.open(io.BytesIO(base64.b64decode(image_data)))
            return image.convert("RGB")
        
        # Handle uploaded images
        # (implementation for file uploads)
        
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
    
    def flush(self):
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_max_memory_allocated(i)
            torch.cuda.reset_max_memory_cached(i)
            torch.cuda.reset_peak_memory_stats(i)
            torch.cuda.reset_accumulated_memory_stats(i)
        gc.collect()