from pydantic import BaseModel

class ImageRequest(BaseModel):
    model: str = 'black-forest-labs_FLUX.1-dev'
    steps: int = 50
    prompt: str
    size: str = '1280x720'
    guidance_scale: float = 5.0

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = None
    stream: bool = False