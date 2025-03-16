from fastapi import APIRouter, HTTPException
from schemas import ChatCompletionRequest
from config import config
import requests
import uuid
import time

router = APIRouter(prefix="/v1")

OLLAMA_BASE_URL = f"http://{config.ollama.host}:{config.ollama.port}"
OLLAMA_CHAT_ENDPOINT = f"{OLLAMA_BASE_URL}/api/chat"

# Chat completions API
#
# POST /v1/chat/completions
# Headers:
#   Content-Type: application/json
# Body:
#   {
#     "model": "<ollama-model-name>",
#     "messages": [
#       {"role": "system|user|assistant", "content": "<message-text>"},
#       ...
#     ],
#     "temperature": <0.0-1.0>,    # Optional (default: 0.7)
#     "top_p": <0.0-1.0>,          # Optional (default: 1.0)
#     "max_tokens": <n>,           # Optional (default: unlimited)
#     "stream": <bool>             # Optional (default: false)
#   }
#
# Example of API call
# curl -X POST -H 'Content-Type: application/json' \
#  -d '{
#    "model": "llama2",
#    "messages": [
#      {"role": "user", "content": "Tell me a joke about AI"}
#    ],
#    "temperature": 0.7
#  }' \
#  http://localhost:8000/v1/chat/completions
#
# Note: Requires running Ollama server with specified model (e.g. `ollama pull llama2`)

@router.post("/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    try:
        # Преобразование запроса в формат Ollama
        ollama_payload = {
            "model": request.model,
            "messages": [m.dict() for m in request.messages],
            "options": {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "num_predict": request.max_tokens
            },
            "stream": request.stream
        }

        response = requests.post(
            OLLAMA_CHAT_ENDPOINT,
            json=ollama_payload,
            timeout=config.ollama.timeout
        )
        response.raise_for_status()
        ollama_response = response.json()

        # Преобразование ответа в формат OpenAI
        openai_response = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": ollama_response["message"]["role"],
                    "content": ollama_response["message"]["content"]
                },
                "finish_reason": "stop" if ollama_response["done"] else "length"
            }],
            "usage": {
                "prompt_tokens": ollama_response["prompt_eval_count"],  # Ollama не предоставляет эту информацию
                "completion_tokens": ollama_response["eval_count"],
                "total_tokens": ollama_response["prompt_eval_count"] + ollama_response["eval_count"]
            }
        }

        return openai_response

    except requests.exceptions.ReadTimeout:
        raise HTTPException(
            status_code=504,
            detail="Ollama response timeout. Consider using a simpler model or shorter prompt"
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ollama API error: {str(e)}"
        )