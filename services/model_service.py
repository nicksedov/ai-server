from huggingface_hub import scan_cache_dir, snapshot_download, HFCacheInfo, HfApi, ModelInfo
from transformers import AutoConfig, AutoTokenizer
import warnings
from config import config
from pathlib import Path
from typing import List, Dict
import shutil
import requests
import logging
import time

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self):
        self.ollama_base = f"http://{config.ollama.host}:{config.ollama.port}"
        self.timeout = config.ollama.timeout

    async def get_all_models(self):
        return self._get_ollama_models() + self._get_huggingface_models()

    async def download_model(self, provider: str, model_id: str, revision: str = "main"):
        if provider == "ollama":
            return await self._download_ollama_model(model_id)
        elif provider == "huggingface":
            return await self._download_huggingface_model(model_id, revision)
        raise ValueError("Unsupported provider")

    async def delete_model(self, provider: str, model_id: str, purge: bool):
        if provider == "ollama":
            return await self._delete_ollama_model(model_id)
        elif provider == "huggingface":
            return await self._delete_huggingface_model(model_id, purge)
        raise ValueError("Unsupported provider")

    async def _download_ollama_model(self, model_id: str):
        response = requests.post(
            f"{self.ollama_base}/api/pull",
            json={"name": model_id},
            timeout=self.timeout,
            stream=False
        )
        response.raise_for_status()
        return {
            "status": "success", 
            "message": f"Model {model_id} downloaded", 
            "model_id": model_id, 
            "provider": "ollama"
        }

    async def _download_huggingface_model(self, model_id: str, revision: str):
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            resume_download=True,
            local_files_only=False
        )
        return {
            "status": "success", 
            "message": f"Model {model_id} downloaded", 
            "model_id": model_id, 
            "provider": "huggingface"
        }

    async def _delete_ollama_model(self, model_id: str):
        response = requests.delete(
            f"{self.ollama_base}/api/delete",
            json={"name": model_id},
            timeout=self.timeout
        )
        response.raise_for_status()
        return {
            "status": "success", 
            "message": f"Model {model_id} deleted", 
            "model_id": model_id, 
            "provider": "ollama"
        }

    async def _delete_huggingface_model(self, model_id: str, purge: bool):
        cache_info = scan_cache_dir()
        deleted = False

        for repo in cache_info.repos:
            if repo.repo_id == model_id and repo.repo_type == "model":
                try:
                    if purge:
                        shutil.rmtree(Path(repo.repo_path))
                    else:
                        for revision in repo.revisions:
                            (revision.snapshot_path / ".lock").unlink(missing_ok=True)
                    deleted = True
                except Exception as e:
                    continue
        return {
            "status": "success" if deleted else "failed",
            "model_id": model_id,
            "provider": "huggingface"
        }
    
    def _get_ollama_models(self) -> List[Dict]:
        models = []
        try:
            response = requests.get(
                f"{self.ollama_base}/api/tags",
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
                    "owned_by": "ollama",
                    "is_chat": True,
                    "is_multimodal": self._is_multimodal(self._get_ollama_tags(model["name"]))
                })

        except Exception as e:
            logger.error(f"Ollama models error: {str(e)}")

        return models

    def _get_huggingface_models(self) -> List[Dict]:
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
                    "owned_by": "huggingface",
                    "is_chat": self._is_huggingface_chat_model(model_id),
                    "is_multimodal": self._is_multimodal(self._get_huggingface_tags(model_id))
                })

        except Exception as e:
            logger.error(f"Huggingface models error: {str(e)}")

        return models

    def _is_huggingface_chat_model(self, model_id: str) -> bool:
        """
        Проверяет, может ли модель использоваться для генерации ответов чат-бота.
        Возвращает True/False.
        """
        try:
            # 1. Проверяем теги репозитория
            tags = [tag.lower() for tag in self._get_huggingface_tags(model_id)]
            has_chat_tags = any(tag in tags for tag in [
                'chat', 'conversational', 'text-generation', 'chat-completion', 'dialog', 'image-text-to-text'
            ])
            if has_chat_tags:
                return True
        except Exception as e:
            warnings.warn(f"Error checking model {model_id}: {str(e)}")
            return False

        # 2. Проверяем наличие чат-шаблона в токенизаторе
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            has_chat_template = tokenizer.chat_template is not None
        except:
            has_chat_template = False
        return has_chat_template

    
        
    def _get_huggingface_tags(self, model_id: str) -> List[str]:
        """Получение тегов модели из Hugging Face Hub"""
        try:
            model_info = HfApi().model_info(model_id)
            return model_info.tags
        except Exception as e:
            logger.warning(f"Can't get tags for {model_id}: {str(e)}")
            return []

    def _get_ollama_tags(self, model_id: str) -> List[str]:
        """Получение тегов модели Ollama через API"""
        try:
            response = requests.post(
                f"{self.ollama_base}/api/show",
                json={"name": model_id},
                timeout=config.ollama.timeout
            )
            response.raise_for_status()
            metadata = response.json()
            
            # Анализ параметров модели
            tags = metadata.get('capabilities')

            family = metadata.get('details', {}).get('family')
            if (family):
                tags.append(family)
            return tags    
        except Exception as e:
            logger.warning(f"Can't get Ollama model tags: {str(e)}")
            return []

    def _is_multimodal(self, tags: List[str]) -> bool:
        return any(tag in tags for tag in ["multimodal", "vision", "vl", "qwen2vl", "image-text-to-text"])