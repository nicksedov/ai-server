from pydantic import BaseModel
from typing import Dict, Optional

class ComponentsHealthResponse(BaseModel):
    api: bool = True
    models_cache: Optional[bool]
    gpu: Optional[Dict]

class HealthResponse(BaseModel):
    status: str = "OK"
    timestamp: str
    components: ComponentsHealthResponse