import requests
from huggingface_hub import scan_cache_dir, HFCacheInfo
from config import config
import logging
import time
from typing import List, Dict

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = f"http://{config.ollama.host}:{config.ollama.port}"

def get_ollama_models() -> List[Dict]:
    models = []
    try:
        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=config.ollama.timeout
        )
        response.raise_for_status()
        ollama_response = response.json()
        
        for model in ollama_response.get("models", []):
            modified_time = int(time.mktime(
                time.strptime(
                    model["modified_at"].split(".")[0], 
                    "%Y-%m-%dT%H:%M:%S"
                )
            )) if "modified_at" in model else int(time.time())
            
            models.append({
                "id": model["name"],
                "created": modified_time,
                "owned_by": "ollama"
            })

    except Exception as e:
        logger.error(f"Ollama models error: {str(e)}")

    return models

def get_huggingface_models() -> List[Dict]:
    models = []
    try:
        cache_info: HFCacheInfo = scan_cache_dir()
        hf_models = {}

        for repo in cache_info.repos:
            if repo.repo_type == "model":
                try:
                    revisions = sorted(
                        list(repo.revisions),
                        key=lambda r: getattr(r, 'last_modified', 0),
                        reverse=True
                    )
                    
                    if not revisions:
                        continue
                    
                    last_rev = revisions[0]
                    last_modified = getattr(last_rev, 'last_modified', None)
                    
                    if isinstance(last_modified, float):
                        created = int(last_modified)
                    elif hasattr(last_modified, 'timestamp'):
                        created = int(last_modified.timestamp())
                    else:
                        created = int(time.time())
                    
                    model_id = repo.repo_id
                    
                    if hasattr(repo, 'repo_name'):
                        model_id = f"{repo.repo_namespace}/{repo.repo_name}" if repo.repo_namespace else repo.repo_name
                    
                    if model_id not in hf_models or created > hf_models[model_id]:
                        hf_models[model_id] = created

                except Exception as e:
                    logger.warning(f"Error processing {repo.repo_id}: {str(e)}")
                    continue

        for model_id, created in hf_models.items():
            models.append({
                "id": model_id,
                "created": created,
                "owned_by": "huggingface"
            })

    except Exception as e:
        logger.error(f"Huggingface models error: {str(e)}")

    return models

def get_all_models() -> List[Dict]:
    return get_ollama_models() + get_huggingface_models()