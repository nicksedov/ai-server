from typing import Annotated
from fastapi import APIRouter, Request, Header  # Базовые компоненты FastAPI
from schemas import ImageRequest  # Pydantic-схема для валидации запросов
from utils import flush, translate_to_english  # Вспомогательные утилиты
from config import config  # Конфигурационные параметры приложения
import torch  # Основная библиотека для работы с нейросетями
from diffusers import FluxPipeline, AutoencoderKL, FluxTransformer2DModel  # Компоненты модели диффузии
from diffusers.image_processor import VaeImageProcessor  # Обработчик изображений для VAE
import datetime
import os

# Инициализация роутера API с префиксом /v1
router = APIRouter(prefix="/v1")

# Эндпоинт для генерации изображений через POST-запрос в формате OpenAI API
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
        body: ImageRequest,  # Тело запроса, валидируемое схемой ImageRequest
        content_language: Annotated[str | None, Header()] = None,  # Язык промпта из заголовка
    ):
    # Извлечение параметров из тела запроса
    ckpt_id = body.model       # Идентификатор модели
    steps = body.steps          # Количество шагов генерации
    prompt = body.prompt        # Текст промпта
    if content_language:
        # Автоматический перевод промпта на английский (модель работает с английским текстом)
        prompt = await translate_to_english(body.prompt, content_language)

    # Этап 1: Подготовка текстовых эмбеддингов --------------------------------
    # Инициализация пайплайна для работы с текстовым кодировщиком
    pipeline = FluxPipeline.from_pretrained(
        ckpt_id,
        transformer=None,     # Не загружаем трансформер для экономии памяти
        vae=None,             # Не загружаем VAE
        device_map="balanced", # Оптимальное распределение по GPU
        max_memory={0: "24GB", 1: "16GB"},  # Лимиты памяти для устройств
        torch_dtype=torch.bfloat16  # Использование экономичного формата данных
    )

    # Генерация текстовых эмбеддингов
    with torch.no_grad():
        print("Encoding prompts.")
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompt, 
            prompt_2=None, 
            max_sequence_length=512  # Максимальная длина текстового контекста
        )
    
    # Очистка памяти от компонентов, связанных с текстовым кодировщиком
    pipeline.text_encoder = None
    pipeline.text_encoder_2 = None
    pipeline.tokenizer = None
    pipeline.tokenizer_2 = None
    del pipeline
    flush()  # Принудительная очистка памяти GPU

    # Этап 2: Генерация латентных представлений ------------------------------
    # Загрузка трансформера для обработки латентных представлений
    transformer = FluxTransformer2DModel.from_pretrained(
        ckpt_id, 
        subfolder="transformer",  # Поддиректория в репозитории модели
        device_map="auto",        # Автоматическое распределение по устройствам
        torch_dtype=torch.bfloat16
    )
    
    # Инициализация пайплайна для работы с трансформером
    pipeline = FluxPipeline.from_pretrained(
        ckpt_id,
        text_encoder=None,    # Текстовый кодировщик уже не нужен
        vae=None,             # VAE пока не загружаем
        transformer=transformer,  # Используем загруженный трансформер
        torch_dtype=torch.bfloat16
    )

    # Парсинг размеров выходного изображения
    size = tuple(map(int, body.size.split('x')))
    width = size[0] if len(size) == 2 and size[0] >= 16 else 1280  # Дефолтная ширина
    height = size[1] if len(size) == 2 and size[1] >= 16 else 768  # Дефолтная высота

    # Генерация латентных представлений изображения
    latents = pipeline(
        prompt_embeds=prompt_embeds,             # Эмбеддинги из первого этапа
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_inference_steps=steps,              # Количество шагов денойзинга
        guidance_scale=3.5,                     # Коэффициент соблюдения промпта
        height=height,
        width=width,
        output_type="latent",                   # Возвращаем сырые латенты
    ).images

    # Очистка памяти от трансформера
    pipeline.transformer = None
    del pipeline
    del transformer
    flush()

    # Этап 3: Декодирование латентов в изображение ---------------------------
    # Загрузка Variational AutoEncoder (VAE) для преобразования латентов
    vae = AutoencoderKL.from_pretrained(
        ckpt_id, 
        subfolder="vae", 
        torch_dtype=torch.bfloat16
    ).to("cuda")  # Явное перемещение на GPU

    vae_scale_factor = 8  # Коэффициент масштабирования для VAE
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    # Декодирование латентов в изображение
    with torch.no_grad():
        print("Running decoding.")
        # Подготовка латентов к декодированию
        latents = FluxPipeline._unpack_latents(latents, height, width, vae_scale_factor)
        latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
        
        # Декодирование с использованием VAE
        image = vae.decode(latents.to(vae.device, dtype=vae.dtype), return_dict=False)[0]
        image = image_processor.postprocess(image, output_type="pil")  # Конвертация в PIL.Image
        
        # Генерация уникального имени файла
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        root_path = config.server.storage_root  # Путь из конфигурации
        filename = f"{root_path}/AI/output/flux/flux_{width}x{height}_{steps}st_{now}.png"
        image[0].save(filename)  # Сохранение изображения на диск

    # Финализация: освобождение ресурсов
    vae.to('cpu')  # Перенос VAE на CPU перед удалением
    del vae
    del latents
    del image
    flush()

    return {"prompt": prompt, "filepath": filename}  # Возврат результата клиенту