from pydantic import BaseModel

class ImageRequest(BaseModel):
    model: str = 'black-forest-labs/FLUX.1-dev'
    steps: int = 50
    prompt: str
    size: str = '1280x720'
    guidance_scale: float = 4.5