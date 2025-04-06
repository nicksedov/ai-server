from fastapi import APIRouter, Request, Header, Depends, HTTPException
from fastapi.responses import FileResponse
from schemas import ImageRequest
from services.image_service import ImageService
from config import config
from models_cache import model_cache
from auth import verify_auth
import logging
import os

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)
image_service = ImageService()

@router.post("/images/generations", 
            dependencies=[Depends(verify_auth)],
            summary="Генерация изображения по текстовому промпту",
            description="""Генерация изображений через диффузионные модели. Особенности:
- Автоматический перевод промптов на английский язык
- Поддержка кастомных размеров через параметр size
- Возвращает URL для скачивания изображения""")
async def generate_image(
    request: Request,
    body: ImageRequest,
    content_language: str = Header(default='auto')
):
    try:
        model_cache.validate_model(body.model)
        result = await image_service.generate_and_save_image(
            model=body.model,
            prompt=body.prompt,
            language=content_language,
            steps=body.steps,
            size=body.size,
            guidance_scale=body.guidance_scale
        )
        return format_image_response(request, result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image generation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Image generation error: {str(e)}"
        )

@router.get("/images/{filename}",
           summary="Получение сгенерированного изображения",
           description="Возвращает ранее сгенерированное изображение по имени файла")
async def get_image(filename: str):
    try:
        image_path = os.path.join(config.server.output_path, filename)
        if not os.path.isfile(image_path):
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(image_path)
    except Exception as e:
        logger.error(f"Image retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

def format_image_response(request: Request, result: dict) -> dict:
    filename = os.path.basename(result["filepath"])
    return {
        "created": int(result["created"]),
        "data": [{
            "url": f"{request.base_url}v1/images/{filename}",
            "revised_prompt": result["prompt"]
        }],
        "model": "black-forest-labs/FLUX.1-dev",
        "parameters": {
            "steps": result.get("steps"),
            "size": result.get("size"),
            "guidance_scale": result.get("guidance_scale")
        }
    }