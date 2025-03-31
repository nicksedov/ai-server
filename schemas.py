from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class ImageRequest(BaseModel):
    model: str = 'black-forest-labs/FLUX.1-dev'
    steps: int = 50
    prompt: str
    size: str = '1280x720'
    guidance_scale: float = 5.5

class ChatMessage(BaseModel):
    role: str = "user"
    content: str = "Назови планеты Солнечной системы"

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 8192
    stream: bool = False

class DownloadRequest(BaseModel):
    provider: str = "huggingface" # "ollama" или "huggingface"
    model_id: str
    revision: str = "main"  # Для Hugging Face

class DeleteRequest(BaseModel):
    provider: str = "huggingface" # "ollama" или "huggingface"
    model_id: str
    purge: bool = True  # Полное удаление для Hugging Face

class CPUInfo(BaseModel):
    usage_percent: float = Field(..., example=24.5)
    cores_physical: int = Field(..., example=8)
    cores_logical: int = Field(..., example=16)
    frequency_current: float = Field(..., example=3600.0)
    frequency_max: float = Field(..., example=4800.0)
    load_1m: Optional[float] = Field(None, example=1.2)
    load_5m: Optional[float] = Field(None, example=1.5)
    load_15m: Optional[float] = Field(None, example=1.3)

class MemoryInfo(BaseModel):
    total: float = Field(..., example=62.8)
    available: float = Field(..., example=32.1)
    used: float = Field(..., example=30.7)
    percent: float = Field(..., example=48.9)
    swap_total: Optional[float] = Field(None, example=8.0)
    swap_used: Optional[float] = Field(None, example=1.5)

class GPUInfo(BaseModel):
    name: str
    memory_total: int
    memory_used: int
    memory_free: int
    utilization: int
    temperature: int
    fan_speed_percent: Optional[int] = None

class SystemInfoResponse(BaseModel):
    timestamp: float
    os: str
    cpu: CPUInfo
    memory: MemoryInfo
    cuda_available: bool
    gpus: List[GPUInfo] = []
    torch_version: str
    python_version: str