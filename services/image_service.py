import torch
from diffusers import FluxPipeline, AutoencoderKL, FluxTransformer2DModel
from diffusers.image_processor import VaeImageProcessor
from utils import translate_to_english
from config import config
import datetime
import os
import gc

class ImageService:
    def __init__(self):
        self.device_map = "balanced"
        self.torch_dtype = torch.bfloat16

    # async def generate_and_save_image(
    #     self,
    #     prompt: str,
    #     language: str,
    #     steps: int,
    #     size: str,
    #     guidance_scale: float
    # ):
    #     # Перевод промпта на английский
    #     translated_prompt = await translate_to_english(prompt, language)
        
    #     pipeline = await self._initialize_pipeline()
    #     prompt_embeds, pooled_prompt_embeds, text_ids = await self._encode_prompt(pipeline, translated_prompt)

    #     # Очистка памяти от компонентов, связанных с текстовым кодировщиком
    #     pipeline.text_encoder = None
    #     pipeline.text_encoder_2 = None
    #     pipeline.tokenizer = None
    #     pipeline.tokenizer_2 = None
    #     del pipeline
    #     self._cleanup()  # Принудительная очистка памяти GPU
        
    #     # Парсинг размера изображения
    #     width, height = map(int, size.split('x'))
        
    #     latents = await self._generate_latents(
    #         pipeline=pipeline,
    #         prompt_embeds=prompt_embeds,
    #         steps=steps,
    #         width=width,
    #         height=height,
    #         guidance_scale=guidance_scale
    #     )
        
    #     image = await self._decode_latents(latents)
    #     filename = self._save_image(image, size, steps)
    #     self._cleanup()
        
    #     return {
    #         "prompt": translated_prompt,
    #         "filepath": filename,
    #         "created": datetime.datetime.now().timestamp(),
    #         "steps": steps,
    #         "size": size,
    #         "guidance_scale": guidance_scale
    #     }

    # async def _initialize_pipeline(self):
    #     return FluxPipeline.from_pretrained(
    #         "black-forest-labs/FLUX.1-dev",
    #         device_map=self.device_map,
    #         max_memory={0: "24GB", 1: "16GB"},
    #         torch_dtype=self.torch_dtype
    #     )

    # async def _encode_prompt(self, pipeline, prompt: str):
    #     with torch.no_grad():
    #         return pipeline.encode_prompt(
    #             prompt=prompt,
    #             prompt_2=None,
    #             max_sequence_length=512
    #         )

    # async def _generate_latents(
    #     self,
    #     pipeline,
    #     prompt_embeds,
    #     steps: int,
    #     width: int,
    #     height: int,
    #     guidance_scale: float
    # ):
    #     transformer = await self._load_transformer()
    #     pipeline.transformer = transformer
        
    #     return pipeline(
    #         prompt_embeds=prompt_embeds,
    #         num_inference_steps=steps,
    #         guidance_scale=guidance_scale,
    #         width=width,
    #         height=height,
    #         output_type="latent"
    #     ).images

    # async def _load_transformer(self):
    #     return FluxTransformer2DModel.from_pretrained(
    #         "black-forest-labs/FLUX.1-dev",
    #         subfolder="transformer",
    #         device_map="auto",
    #         torch_dtype=self.torch_dtype
    #     )

    # async def _decode_latents(self, latents):
    #     vae = await self._load_vae()
    #     image_processor = VaeImageProcessor(vae_scale_factor=8)
    #     return image_processor.postprocess(
    #         vae.decode(latents)[0],
    #         output_type="pil"
    #     )

    # async def _load_vae(self):
    #     vae = AutoencoderKL.from_pretrained(
    #         "black-forest-labs/FLUX.1-dev",
    #         subfolder="vae",
    #         torch_dtype=self.torch_dtype
    #     ).to("cuda")
    #     return vae

    # def _save_image(self, image, size: str, steps: int) -> str:
    #     now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #     filename = (
    #         f"{config.server.storage_root}/AI/output/flux/"
    #         f"flux_{size}_{steps}st_{now}.png"
    #     )
    #     image[0].save(filename)
    #     return filename

    # def _cleanup(self):
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     torch.cuda.reset_max_memory_allocated()
    #     torch.cuda.reset_peak_memory_stats()

    async def generate_and_save_image(
        self,
        model: str,
        prompt: str,
        language: str,
        steps: int,
        size: str,
        guidance_scale: float
    ):
        if language and language != 'en':
            # Автоматический перевод промпта на английский (модель работает с английским текстом)
            prompt = await translate_to_english(prompt, language)

        # Этап 1: Подготовка текстовых эмбеддингов --------------------------------
        # Инициализация пайплайна для работы с текстовым кодировщиком
        pipeline = FluxPipeline.from_pretrained(
            model,
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
        self.flush()  # Принудительная очистка памяти GPU

        # Этап 2: Генерация латентных представлений ------------------------------
        # Загрузка трансформера для обработки латентных представлений
        transformer = FluxTransformer2DModel.from_pretrained(
            model, 
            subfolder="transformer",  # Поддиректория в репозитории модели
            device_map="auto",        # Автоматическое распределение по устройствам
            torch_dtype=torch.bfloat16
        )
        
        # Инициализация пайплайна для работы с трансформером
        pipeline = FluxPipeline.from_pretrained(
            model,
            text_encoder=None,    # Текстовый кодировщик уже не нужен
            vae=None,             # VAE пока не загружаем
            transformer=transformer,  # Используем загруженный трансформер
            torch_dtype=torch.bfloat16
        )

        # Парсинг размеров выходного изображения
        size = tuple(map(int, size.split('x')))
        width = size[0] if len(size) == 2 and size[0] >= 16 else 1280  # Дефолтная ширина
        height = size[1] if len(size) == 2 and size[1] >= 16 else 768  # Дефолтная высота

        # Генерация латентных представлений изображения
        latents = pipeline(
            prompt_embeds=prompt_embeds,             # Эмбеддинги из первого этапа
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=steps,              # Количество шагов денойзинга
            guidance_scale=guidance_scale,          # Коэффициент соблюдения промпта
            height=height,
            width=width,
            output_type="latent",                   # Возвращаем сырые латенты
        ).images

        # Очистка памяти от трансформера
        pipeline.transformer = None
        del pipeline
        del transformer
        self.flush()

        # Этап 3: Декодирование латентов в изображение ---------------------------
        # Загрузка Variational AutoEncoder (VAE) для преобразования латентов
        vae = AutoencoderKL.from_pretrained(
            model, 
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
            filename = f"flux_{width}x{height}_{steps}st_{now}.png"
            full_path = os.path.join(root_path, "output", filename)
            image[0].save(full_path)  # Сохранение изображения на диск

        # Финализация: освобождение ресурсов
        vae.to('cpu')  # Перенос VAE на CPU перед удалением
        del vae
        del latents
        del image
        self.flush()

        return {
            "prompt": prompt,
            "filepath": filename,
            "created": datetime.datetime.now().timestamp(),
            "steps": steps,
            "size": size,
            "guidance_scale": guidance_scale
        }  # Возврат результата клиенту

    def flush(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()