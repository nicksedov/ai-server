from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from routes import images, chat, models, system, ui
from config import config
from models_cache import model_cache
import uvicorn
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Initializing model cache...")
    await model_cache.refresh()
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown logic (при необходимости)
    logger.info("Application shutting down")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    # Используем параметры из конфига
    openapi_config = config.openapi
    
    openapi_schema = get_openapi(
        title=openapi_config.title,
        version=openapi_config.version,
        summary=openapi_config.summary,
        description=openapi_config.description,
        contact=openapi_config.contact,
        license_info=openapi_config.license_info,
        routes=app.routes,
    )

    # Добавляем security схемы из конфига
    openapi_schema["components"]["securitySchemes"] = openapi_config.security_schemes
    openapi_schema["security"] = openapi_config.security
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

app.include_router(images.router)
app.include_router(chat.router)
app.include_router(models.router)
app.include_router(system.router)
app.include_router(ui.router)

if __name__ == "__main__":
    logging.basicConfig(
        filename=config.logging.file_path, 
        level=config.logging.level,
        format=config.logging.format,
        datefmt=config.logging.datefmt
    )
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port
    )