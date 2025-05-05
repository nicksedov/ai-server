
from fastapi import HTTPException, Header, Security
from fastapi.security import APIKeyHeader
from config import config
from datetime import datetime

api_key_header = APIKeyHeader(
    name="Authorization",
    scheme_name="Bearer",
    description="Введите ключ доступа к API",
    auto_error=False
)

async def verify_auth(api_key: str = Security(api_key_header)):
    if not config.server.api_keys:
        return  # Проверка отключена
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Authorization header is missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    scheme, _, token = api_key.partition(" ")
    if scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication scheme",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    now = datetime.now().astimezone()
    valid_key = None
    
    for key in config.server.api_keys:
        if key.key == token:
            if not key.enabled:
                raise HTTPException(
                    status_code=401,
                    detail="API key disabled",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            if key.valid_from and key.valid_from > now:
                raise HTTPException(
                    status_code=401,
                    detail=f"API key not active until {key.valid_from}",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            if key.valid_until and key.valid_until < now:
                raise HTTPException(
                    status_code=401,
                    detail=f"API key expired at {key.valid_until}",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            valid_key = key
            break
    
    if not valid_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )