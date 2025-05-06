from pydantic import BaseModel, Field
from typing import Union, List, Dict, Optional

class ChatContentItem(BaseModel):
    type: str  # text/image
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None

class ChatMessage(BaseModel):
    role: str = "user"
    content: Union[str, List[ChatContentItem]]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 8192
    stream: bool = False
