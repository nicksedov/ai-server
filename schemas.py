from pydantic import BaseModel, Field

class ImageRequest(BaseModel):
    model: str = 'black-forest-labs/FLUX.1-dev'
    steps: int = 50
    prompt: str
    size: str = '1280x720'
    guidance_scale: float = 5.0

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