from pydantic import BaseModel
from typing import List

class DownloadRequest(BaseModel):
    provider: str = "huggingface"
    model_id: str
    revision: str = "main"

class DownloadResponse(BaseModel):
    status: str
    message: str
    provider: str
    model_id: str

class DeleteRequest(BaseModel):
    provider: str = "huggingface"
    model_id: str
    purge: bool = True

class DeleteResponse(BaseModel):
    status: str
    message: str
    provider: str
    model_id: str
    purged: bool = True

class ModelResponse(BaseModel):
    id: str
    created: int
    owned_by: str
    object: str = "model"

class ModelsListResponse(BaseModel):
    object: str = "list"
    data: List['ModelResponse']

# Обновляем forward reference после объявления ModelsListResponse
ModelResponse.update_forward_refs()