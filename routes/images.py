from fastapi import APIRouter, Request
from schemas import ImageRequestBody
from utils import flush, translate_to_english
from config import config
import torch
from diffusers import FluxPipeline, AutoencoderKL, FluxTransformer2DModel
from diffusers.image_processor import VaeImageProcessor
import datetime
import os

router = APIRouter(prefix="/v1")

# Image generation API
#
# POST /v1/images/generations
# Headers:
#   Content-Type: application/json
#   Content-Language: <prompt language, e.g. 'ru'. Default is 'en'>
# Body:
#   {
#     "model": "<model>",
#     "steps": <n>,
#     "prompt": "<text>", 
#     "size": "<width>x<height>", 
#   }
# Example of API call
# curl -X POST -H 'Content-Type: application/json' -H 'Content-Language: ru'\
#  -d '{"model":"black-forest-labs/FLUX.1-dev", "steps":50, "prompt":"Красная Шапочка встречеет волка", "size": "512x512"}' \
#  http://localhost:8000/v1/images/generations
#
@router.post("/images/generations")
async def generate_image(
        request: Request, 
        body: ImageRequestBody,
    ):
    content_language = request.headers.get('Content-Language')
    ckpt_id = body.model
    steps = body.steps
    prompt = body.prompt
    if content_language:
        prompt = await translateToEnglish(body.prompt, content_language)

    # Инициализация первого пайплайна
    pipeline = FluxPipeline.from_pretrained(
        ckpt_id,
        transformer=None,
        vae=None,
        device_map="balanced",
        max_memory={0: "24GB", 1: "16GB"},
        torch_dtype=torch.bfloat16
    )

    with torch.no_grad():
        print("Encoding prompts.")
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompt, prompt_2=None, max_sequence_length=512
        )
    
    # Очистка первого пайплайна
    pipeline.text_encoder = None
    pipeline.text_encoder_2 = None
    pipeline.tokenizer = None
    pipeline.tokenizer_2 = None
    del pipeline
    flush()

    # Загрузка трансформера
    transformer = FluxTransformer2DModel.from_pretrained(
        ckpt_id, 
        subfolder="transformer",
        device_map="auto",
        max_memory={0: "24GB", 1: "16GB"},
        torch_dtype=torch.bfloat16
    )
    
    # Инициализация второго пайплайна
    pipeline = FluxPipeline.from_pretrained(
        ckpt_id,
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
        vae=None,
        transformer=transformer,
        torch_dtype=torch.bfloat16
    )

    # Генерация латентов
    size = tuple(map(int, body.size.split('x')))
    width = size[0] if len(size) == 2 and size[0] >= 16 else 1280
    height = size[1] if len(size) == 2 and size[1] >= 16 else 768

    latents = pipeline(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_inference_steps=steps,
        guidance_scale=3.5,
        height=height,
        width=width,
        output_type="latent",
    ).images

    # Очистка второго пайплайна и трансформера
    pipeline.transformer = None
    del pipeline
    del transformer
    flush()

    # Загрузка и использование VAE
    vae = AutoencoderKL.from_pretrained(ckpt_id, subfolder="vae", torch_dtype=torch.bfloat16).to("cuda")
    vae_scale_factor = 8
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    with torch.no_grad():
        print("Running decoding.")
        latents = FluxPipeline._unpack_latents(latents, height, width, vae_scale_factor)
        latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
        image = vae.decode(latents.to(vae.device, dtype=vae.dtype), return_dict=False)[0]
        image = image_processor.postprocess(image, output_type="pil")
        
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        root_path = config.server.storage_root
        filename = f"{root_path}/AI/output/flux/flux_{width}x{height}_{steps}st_{now}.png"
        image[0].save(filename)

    # Финализация очистки
    vae.to('cpu')  # Перемещаем VAE на CPU перед удалением
    del vae
    del latents
    del image
    flush()

    return {"prompt": prompt, "filepath": filename}