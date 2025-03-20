from fastapi import APIRouter, HTTPException,  status
from huggingface_hub import scan_cache_dir, HFCacheInfo,  snapshot_download, HfApi
from config import config
from schemas import DownloadRequest, DeleteRequest
from concurrent.futures import ThreadPoolExecutor
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
@router.get("/models")
async def list_models():
    models_list = []
    
    try:
        # Обработка моделей Ollama
        ollama_response = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=config.ollama.timeout
        ).json()
        
        for model in ollama_response.get("models", []):
            modified_time = int(time.mktime(
                time.strptime(
                    model["modified_at"].split(".")[0], 
                    "%Y-%m-%dT%H:%M:%S"
                )
            )) if "modified_at" in model else int(time.time())
            
            models_list.append({
                "id": model["name"],
                "object": "model",
                "created": modified_time,
                "owned_by": "ollama"
            })

    except Exception as e:
        logger.error(f"Ollama models error: {str(e)}")

    try:
        # Обработка моделей Huggingface
        cache_info: HFCacheInfo = scan_cache_dir()
        hf_models = {}

        for repo in cache_info.repos:
            if repo.repo_type == "model":
                try:
                    # Безопасная обработка ревизий
                    revisions = sorted(
                        list(repo.revisions),
                        key=lambda r: getattr(r, 'last_modified', 0),
                        reverse=True
                    )
                    
                    if not revisions:
                        continue
                    
                    last_rev = revisions[0]
                    
                    # Получаем временную метку
                    last_modified = getattr(last_rev, 'last_modified', None)
                    
                    if isinstance(last_modified, float):
                        created = int(last_modified)
                    elif hasattr(last_modified, 'timestamp'):
                        created = int(last_modified.timestamp())
                    else:
                        created = int(time.time())  # Fallback
                    
                    # Формируем идентификатор модели
                    model_id = repo.repo_id
                    
                    # Используем имя из конфига репозитория
                    if hasattr(repo, 'repo_name'):
                        model_id = f"{repo.repo_namespace}/{repo.repo_name}" if repo.repo_namespace else repo.repo_name
                    
                    # Дедупликация
                    if model_id not in hf_models or created > hf_models[model_id]:
                        hf_models[model_id] = created

                except Exception as e:
                    logger.warning(f"Error processing {repo.repo_id}: {str(e)}", exc_info=True)
                    continue

        # Добавляем модели в список
        for model_id, created in hf_models.items():
            models_list.append({
                "id": model_id,
                "object": "model",
                "created": created,
                "owned_by": "huggingface"
            })

    except Exception as e:
        logger.error(f"Huggingface models error: {str(e)}", exc_info=True)

    return {
        "object": "list",
        "data": sorted(models_list, key=lambda x: x["created"], reverse=True)
    }

# Добавляем роутер
@router.post("/models/download", status_code=status.HTTP_202_ACCEPTED)
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

@router.delete("/models/delete", status_code=status.HTTP_202_ACCEPTED)
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
                cache_dir=config.huggingface.cache_dir,
                resume_download=True,
                local_files_only=False,
                token=config.huggingface.token,
            )
        )

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