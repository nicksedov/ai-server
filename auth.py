
from fastapi import HTTPException, Header, Security
from fastapi.security import APIKeyHeader
from config import config

# Создаем схему авторизации
api_key_header = APIKeyHeader(
    name="Authorization",
    scheme_name="Bearer",
    description="Введите ключ доступа к API",
    auto_error=False
)

async def verify_auth(api_key: str = Security(api_key_header)):
    if config.server.api_key is None:
        return  # Проверка отключена
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Authorization header is missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Проверяем формат Bearer
    scheme, _, token = api_key.partition(" ")
    if scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication scheme",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if token != config.server.api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )
