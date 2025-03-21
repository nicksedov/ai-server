from pydantic import BaseModel, Field

class ImageRequest(BaseModel):
    model: str = 'black-forest-labs/FLUX.1-dev'
    steps: int = 50
    prompt: str
    size: str = '1280x720'
    guidance_scale: float = 5.0

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(
        default_factory=lambda: [
            ChatMessage(
                role="system", 
                content="Ты полезный ИИ-помощник. Внимательно следуй инструкциям пользователя."
            ),
            ChatMessage(
                role="user",
                content="Назови планеты Солнечной системы"
            )
        ]
    )
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = None
    stream: bool = False

class DownloadRequest(BaseModel):
    provider: str = "huggingface" # "ollama" или "huggingface"
    model_id: str
    revision: str = "main"  # Для Hugging Face

class DeleteRequest(BaseModel):
    provider: str = "huggingface" # "ollama" или "huggingface"
    model_id: str
    purge: bool = True  # Полное удаление для Hugging Face