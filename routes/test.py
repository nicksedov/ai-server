from fastapi import APIRouter, HTTPException,  status
from huggingface_hub import scan_cache_dir, HFCacheInfo,  snapshot_download, HfApi
from concurrent.futures import ThreadPoolExecutor
import requests
import asyncio
import time
import logging
import shutil

cache_info = scan_cache_dir()
deleted = False
model_id = 'Lightricks/LTX-Video'
revision = "main"
purge = True

for repo in cache_info.repos:
    if repo.repo_id == model_id:
        for rev in repo.revisions:
            try:
                # Удаляем все файлы ревизии
                if purge:
                    shutil.rmtree(rev.snapshot_path)
                else:
                    # Помечаем для автоматической очистки
                    (rev.snapshot_path / ".lock").unlink(missing_ok=True)
                
                # Удаляем информацию о ревизии
                rev.blobs_path.unlink(missing_ok=True)
                rev.raw_path.unlink(missing_ok=True)
                
                deleted = True
                logger.info(f"Deleted revision {rev.revision} for {model_id}")

            except Exception as e:
                logger.warning(f"Error deleting {rev.revision}: {str(e)}")
                continue
        