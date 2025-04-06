from pydantic import BaseModel
from typing import Optional, List, Dict

class TorchInfoResponse(BaseModel):
    torch_version: str
    cuda_available: bool
    cuda_version: Optional[str]
    cudnn_version: Optional[int]
    current_device: Optional[str]
    device_properties: Optional[Dict]
    memory_allocated: Optional[int]
    memory_reserved: Optional[int]
    devices: List[Dict]