from pydantic import BaseModel
from typing import List, Dict, Optional

class OllamaInfoResponse(BaseModel):
    status: str
    version: Optional[str] = None
    models: List[Dict] = []
    gpu_support: bool = False
    error: Optional[str] = None
    details: Dict = {}