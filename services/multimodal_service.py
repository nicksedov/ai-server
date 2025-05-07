from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
import base64
import io
import logging
import uuid
import time

logger = logging.getLogger(__name__)

class MultimodalService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loaded_models = {}

    async def process_request(self, request, provider):
        model_id = request.model
        if model_id not in self.loaded_models:
            self._load_model(model_id)

        processor = self.loaded_models[model_id]["processor"]
        model = self.loaded_models[model_id]["model"]
        
        # Process messages
        inputs = await self._prepare_inputs(processor, request.messages)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)
        
        generated_text = processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        return self._format_response(request, generated_text)

    def _load_model(self, model_id):
        logger.info(f"Loading multimodal model: {model_id}")
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.loaded_models[model_id] = {
            "processor": processor,
            "model": model
        }

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
            max_length=2048,
        ).to(self.device)

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