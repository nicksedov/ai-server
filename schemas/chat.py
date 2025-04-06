from pydantic import BaseModel, Field

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