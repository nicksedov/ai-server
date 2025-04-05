import os
import yaml
import argparse
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from typing import List, Dict


class OpenAPIConfig(BaseModel):
    title: str = "API Server"
    version: str = "1.0.0"
    summary: str = "API Server Documentation"
    description: str = ""
    contact: Optional[Dict[str, str]] = None
    license_info: Optional[Dict[str, str]] = None
    security_schemes: Dict = {
        "Bearer": {
            "type": "http",
            "scheme": "bearer"
        }
    }
    security: List[Dict] = [{"Bearer": []}]

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    output_path: str = "/tmp"
    api_key: str = None

class OllamaConfig(BaseModel):
    host: str = "localhost"
    port: int = 11434
    timeout: int = Field(300, ge=10, le=600)
    default_model: str

class AppConfig(BaseModel):
    server: ServerConfig
    ollama: OllamaConfig
    openapi: OpenAPIConfig

def load_config(config_path: Optional[Path] = None) -> AppConfig:
    # Поиск конфига по умолчанию
    default_path = Path(__file__).parent.parent / "config-default.yaml"
    
    # Определение пути к конфигу
    config_file = config_path or default_path
    
    config_data = {}
    if config_file.exists():
        with open(config_file) as f:
            config_data = yaml.safe_load(f)
    else:
        if config_path:  # Явно указанный конфиг не найден
            raise FileNotFoundError(f"Config file not found: {config_path}")
        # Для дефолтного конфига просто используем значения по умолчанию

    # Переопределение переменными окружения
    env_overrides = {
        "server.host": os.getenv("SERVER_HOST"),
        "server.port": os.getenv("SERVER_PORT"),
        "server.output_path": os.getenv("SERVER_OUTPUT_PATH"),
        "ollama.host": os.getenv("OLLAMA_HOST"),
        "ollama.port": os.getenv("OLLAMA_PORT"),
        "ollama.timeout": os.getenv("OLLAMA_TIMEOUT"),
        "ollama.default_model": os.getenv("OLLAMA_DEFAULT_MODEL"),
    }

    for key, value in env_overrides.items():
        if value is not None:
            section, param = key.split(".")
            config_data.setdefault(section, {})[param] = value

    return AppConfig(**config_data)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", 
                      type=Path,
                      default=Path("config-default.yaml"),
                      help="Path to config file")
    return parser.parse_args()

# Инициализация конфига при импорте
args = parse_args()
try:
    config = load_config(args.config)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)