from pydantic import BaseModel

class ImageRequestBody(BaseModel):
    model: str
    steps: int
    prompt: str
    size: str

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