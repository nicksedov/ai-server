import torch
from diffusers import FluxPipeline, AutoencoderKL, FluxTransformer2DModel
from diffusers.image_processor import VaeImageProcessor
from typing import Dict
from .translation_service import translate_to_english
from config import config
import datetime
import os
import gc

FLUX_VAE_SCALE_FACTOR: int = 8

class ImageService:
    def __init__(self):
        self.device_map = "balanced"
        self.torch_dtype = torch.bfloat16

    async def generate_and_save_image(
        self,
        model: str,
        prompt: str,
        language: str,
        steps: int,
        size: str,
        guidance_scale: float
    ):
        # Этап 0: Перевод промпта (если требуется)
        if language and language != 'en':
            prompt = await translate_to_english(prompt, language)

        # Этап 1: Подготовка текстовых эмбеддингов
        embeddings = await self.prepare_text_embeddings(model, prompt)
        
        # Этап 2: Генерация латентных представлений
        width, height = self.parse_size(size)
        latents = await self.generate_latents(
            model=model,
            embeddings=embeddings,
            steps=steps,
            width=width,
            height=height,
            guidance_scale=guidance_scale
        )
        
        # Этап 3: Декодирование и сохранение
        result = await self.decode_and_save_image(
            model=model,
            latents=latents,
            width=width,
            height=height
        )
        embeddings = None

        return {
            "prompt": prompt,
            "filepath": result["filename"],
            "created": datetime.datetime.now().timestamp(),
            "steps": steps,
            "size": f"{width}x{height}",
            "guidance_scale": guidance_scale
        }

    #--------------------------------------------------------------
    # Вспомогательные методы
    #--------------------------------------------------------------
    
    async def prepare_text_embeddings(self, model: str, prompt: str):
        """Этап 1: Подготовка текстовых эмбеддингов"""
        pipeline = FluxPipeline.from_pretrained(
            model,
            transformer=None,
            vae=None,
            device_map=self.device_map,
            max_memory={0: "24GB", 1: "16GB"},
            torch_dtype=self.torch_dtype
        )

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                max_sequence_length=512
            )
        
        # Очистка ресурсов
        pipeline.text_encoder = None
        pipeline.text_encoder_2 = None
        pipeline.tokenizer = None
        pipeline.tokenizer_2 = None
        del pipeline
        self.flush()

        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "text_ids": text_ids
        }

    async def generate_latents(
        self,
        model: str,
        embeddings: Dict[str, torch.Tensor],
        steps: int,
        width: int,
        height: int,
        guidance_scale: float
    ):
        """Этап 2: Генерация латентных представлений"""
        transformer = FluxTransformer2DModel.from_pretrained(
            model,
            subfolder="transformer",
            device_map="auto",
            torch_dtype=self.torch_dtype
        )

        pipeline = FluxPipeline.from_pretrained(
            model,
            text_encoder=None,
            vae=None,
            transformer=transformer,
            torch_dtype=self.torch_dtype
        )

        with torch.no_grad():
            latents = pipeline(
                prompt_embeds=embeddings["prompt_embeds"],
                pooled_prompt_embeds=embeddings["pooled_prompt_embeds"],
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                output_type="latent",
            ).images

        # Очистка ресурсов
        pipeline.transformer = None
        del pipeline
        del transformer
        self.flush()

        return latents

    async def decode_and_save_image(
        self,
        model: str,
        latents: torch.Tensor,
        width: int,
        height: int
    ):
        """Этап 3: Декодирование и сохранение изображения"""
        vae = AutoencoderKL.from_pretrained(
            model,
            subfolder="vae",
            torch_dtype=self.torch_dtype
        ).to("cuda")

        image_processor = VaeImageProcessor(vae_scale_factor=FLUX_VAE_SCALE_FACTOR)

        with torch.no_grad():
            latents = FluxPipeline._unpack_latents(latents, height, width, FLUX_VAE_SCALE_FACTOR)
            latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
            
            image = vae.decode(latents.to(vae.device, dtype=vae.dtype), return_dict=False)[0]
            image = image_processor.postprocess(image, output_type="pil")
            
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"flux_{width}x{height}_{now}.png"
            full_path = os.path.join(config.server.output_path, filename)
            image[0].save(full_path)

        # Очистка ресурсов
        vae.to('cpu')
        del vae
        del latents
        del image_processor
        del image
        self.flush()

        return {"filename": filename}

    def parse_size(self, size_str: str) -> tuple[int, int]:
        """Парсинг размера изображения"""
        try:
            width, height = map(int, size_str.split('x'))
            return (
                width if width >= 16 else 1280,
                height if height >= 16 else 768
            )
        except:
            return (1280, 768)

    def flush(self):
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_max_memory_allocated(i)
            torch.cuda.reset_max_memory_cached(i)
            torch.cuda.reset_peak_memory_stats(i)
            torch.cuda.reset_accumulated_memory_stats(i)
        gc.collect()
