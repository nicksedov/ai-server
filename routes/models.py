from fastapi import APIRouter, HTTPException, Depends, status
from huggingface_hub import scan_cache_dir, HFCacheInfo,  snapshot_download, HfApi
from config import config
from auth import verify_auth
from models_cache import model_cache
from services.models_service import get_all_models
from schemas import DownloadRequest, DeleteRequest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
import asyncio
import time
import logging
import shutil

OLLAMA_BASE_URL = f"http://{config.ollama.host}:{config.ollama.port}"

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)

# Models list API
#
# GET /v1/models
#
# Example of API call
# curl http://localhost:8000/v1/models
@router.get("/models", dependencies=[Depends(verify_auth)])
async def list_models():
    return {
        "object": "list",
        "data": sorted(
            [{"id": m["id"], "object": "model", "created": m["created"], "owned_by": m["owned_by"]} 
             for m in await asyncio.to_thread(get_all_models)],
            key=lambda x: x["created"], 
            reverse=True
        )
    }

@router.delete("/models", dependencies=[Depends(verify_auth)], status_code=status.HTTP_202_ACCEPTED)
async def delete_model(request: DeleteRequest):
    if request.provider == "ollama":
        return await handle_ollama_delete(request)
    elif request.provider == "huggingface":
        return await handle_huggingface_delete(request)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid provider. Supported: 'ollama', 'huggingface'"
        )

# Добавляем роутер
@router.post("/models/download", dependencies=[Depends(verify_auth)], status_code=status.HTTP_202_ACCEPTED)
async def download_model(request: DownloadRequest):
    if request.provider == "ollama":
        return await handle_ollama_download(request)
    elif request.provider == "huggingface":
        return await handle_huggingface_download(request)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid provider. Supported: 'ollama', 'huggingface'"
        )

async def handle_ollama_download(request: DownloadRequest):
    try:
        # Асинхронная отправка запроса через пул потоков
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            response = await loop.run_in_executor(
                pool,
                lambda: requests.post(
                    f"{OLLAMA_BASE_URL}/api/pull",
                    json={"name": request.model_id},
                    timeout=config.ollama.timeout,
                    stream=True
                )
            )
            response.raise_for_status()

            # Чтение потока ответа
            for line in response.iter_lines():
                # Обработка прогресса загрузки
                logger.debug(f"Ollama progress: {line.decode()}")

        await model_cache.refresh()
        return {
            "status": "success",
            "message": f"Model {request.model_id} downloaded",
            "provider": "ollama"
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama download error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Error downloading model from Ollama: {str(e)}"
        )

async def handle_huggingface_download(request: DownloadRequest):
    try:
        # Используем ThreadPoolExecutor для синхронных операций
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,  # Используем дефолтный executor
            lambda: snapshot_download(
                repo_id=request.model_id,
                revision=request.revision,
                resume_download=True,
                local_files_only=False,
                #token=config.huggingface.token, #uncomment if necessary
            )
        )

        await model_cache.refresh()
        return {
            "status": "success",
            "message": f"Model {request.model_id} downloaded",
            "provider": "huggingface"
        }

    except Exception as e:
        logger.error(f"Huggingface download error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error downloading model from Huggingface: {str(e)}"
        )


async def handle_ollama_delete(request: DeleteRequest):
    try:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            response = await loop.run_in_executor(
                pool,
                lambda: requests.delete(
                    f"{OLLAMA_BASE_URL}/api/delete",
                    json={"name": request.model_id},
                    timeout=config.ollama.timeout
                )
            )
            response.raise_for_status()
            
        await model_cache.refresh()
        return {
            "status": "success",
            "message": f"Model {request.model_id} deleted",
            "provider": "ollama"
        }

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {request.model_id} not found in Ollama"
            )
        logger.error(f"Ollama delete error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Error deleting model from Ollama: {str(e)}"
        )

async def handle_huggingface_delete(request: DeleteRequest):
    try:
        loop = asyncio.get_event_loop()
        deleted = await loop.run_in_executor(
            None,
            lambda: delete_hf_model_completely(request.model_id, request.purge)
        )

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {request.model_id} not found in cache"
            )

        await model_cache.refresh()
        return {
            "status": "success",
            "message": f"Model {request.model_id} completely deleted{' with purge' if request.purge else ''}",
            "provider": "huggingface"
        }

    except Exception as e:
        logger.error(f"Huggingface delete error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting model from Huggingface: {str(e)}"
        )

def delete_hf_model_completely(model_id: str, purge: bool) -> bool:
    cache_info = scan_cache_dir()
    deleted = False

    for repo in cache_info.repos:
        if repo.repo_id == model_id and repo.repo_type == "model":
            try:
                repo_path = Path(repo.repo_path)
                
                if purge:
                    # Полное удаление всей директории модели
                    shutil.rmtree(repo_path)
                    logger.info(f"Purged model directory: {repo_path}")
                else:
                    # Мягкое удаление через официальный метод
                    for revision in repo.revisions:
                        # Помечаем для удаления в следующей очистке
                        (revision.snapshot_path / ".lock").unlink(missing_ok=True)
                        revision.blobs_path.unlink(missing_ok=True)
                        revision.raw_path.unlink(missing_ok=True)
                    logger.info(f"Marked for deletion: {model_id}")

                deleted = True

            except Exception as e:
                logger.error(f"Error deleting {model_id}: {str(e)}", exc_info=True)
                continue

    return deleted