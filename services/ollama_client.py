import requests
from config import config

class OllamaClient:
    def __init__(self):
        self.base_url = f"http://{config.ollama.host}:{config.ollama.port}"
        self.timeout = config.ollama.timeout

    async def chat(self, payload: dict):
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()