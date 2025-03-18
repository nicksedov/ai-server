from fastapi import APIRouter, HTTPException
from huggingface_hub import scan_cache_dir, HFCacheInfo
from config import config
import requests
import time
import logging

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
                    # model_id = repo.repo_id.replace("/", "_")  # Альтернатива для совместимости
                    
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