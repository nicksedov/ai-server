from fastapi import APIRouter, HTTPException, Depends, status
from schemas.models import DownloadRequest, DownloadResponse, DeleteRequest, DeleteResponse, ModelResponse, ModelsListResponse
from services.model_service import ModelService
from config import config
from models_cache import model_cache
from auth import verify_auth
import logging
import asyncio

router = APIRouter(prefix="/v1", tags=['models'])
logger = logging.getLogger(__name__)
model_service = ModelService()

@router.get("/models", 
           response_model=ModelsListResponse,
           summary="Получение списка доступных моделей",
           description="""Возвращает список всех доступных моделей из всех провайдеров (Ollama и Hugging Face).
           При передаче параметра chat=true возвращает только чат-оптимизированные модели""")
async def list_models(chat: bool = False):
    """Get list of available models from all providers"""
    try:
        models = model_cache.get_all_models()
        return format_models_response(models, chat)
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to retrieve models list"
        )
        
@router.post("/models/download", 
            dependencies=[Depends(verify_auth)],
            response_model=DownloadResponse,
            status_code=status.HTTP_202_ACCEPTED,
            summary="Загрузка модели из указанного провайдера",
            description="""Инициирует процесс загрузки модели. Поддерживаемые провайдеры:
- ollama: модели из репозитория Ollama
- huggingface: модели с Hugging Face Hub""")
async def download_model(request: DownloadRequest):
    """Download model from specified provider"""
    try:
        result = await model_service.download_model(
            request.provider,
            request.model_id,
            request.revision
        )
        await model_cache.refresh()
        return format_download_response(result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Model download failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model download failed: {str(e)}"
        )

@router.delete("/models", 
             dependencies=[Depends(verify_auth)],
             response_model=DeleteResponse,
             status_code=status.HTTP_202_ACCEPTED,
             summary="Удаление модели из системы",
             description="""Удаляет указанную модель из кеша. Для моделей Hugging Face:
- purge=True: полное удаление модели
- purge=False: пометка для отложенной очистки при следующей сборке мусора""")
async def delete_model(request: DeleteRequest):
    """Delete model from specified provider"""
    try:
        result = await model_service.delete_model(
            request.provider,
            request.model_id,
            request.purge
        )
        await model_cache.refresh()
        return format_delete_response(result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Model deletion failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model deletion failed: {str(e)}"
        )

def format_models_response(models: list, chat: bool) -> ModelsListResponse:  # Добавляем параметр
    filtered_models = [
        m for m in models 
        if not chat or (chat and m["is_chat"])  # Фильтрация по is_chat при chat=true
    ]
    
    return ModelsListResponse(
        data=sorted(
            [
                ModelResponse(
                    id=m["id"],
                    created=m["created"],
                    owned_by=m["owned_by"],
                    is_chat=m["is_chat"],
                    is_multimodal=m["is_multimodal"]
                ) for m in filtered_models  # Используем отфильтрованный список
            ],
            key=lambda x: x.created,
            reverse=True
        )
    )

def format_download_response(result: dict) -> DownloadResponse:
    return DownloadResponse(
        status=result.get("status", "success"),
        message=result.get("message", ""),
        provider=result.get("provider", "unknown"),
        model_id=result.get("model_id", ""),
    )

def format_delete_response(result: dict) -> DeleteResponse:
    return DeleteResponse(
        status=result.get("status", "success"),
        message=result.get("message", ""),
        provider=result.get("provider", "unknown"),
        model_id=result.get("model_id", ""),
        purged=result.get("purge", False),
    )