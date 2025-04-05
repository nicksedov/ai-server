from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from routes import images, chat, models, system
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

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    description = """
    # Key Information:
    - Used exclusively for owner's personal purposes
    - Testing and deployment of Language Learning Models (LLMs)
    - Not intended for public use
    - Access strictly limited to authorized users

    # Technical Details:
    - Integrated with Ollama for model operations
    - Huggingface-hosted models support
    - OpenAI API format compatible
    """

    openapi_schema = get_openapi(
        title="LLM Server API",
        version="1.0.0",
        summary="Private LLM model management server",
        description=description,
        contact={
            "name": "Nikolay Sedov",
            "url": "https://nicksedov.github.io/",
        },
        license_info={
            "name": "PRIVATE USE ONLY"
        },
        routes=app.routes,
    )

    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    
    openapi_schema["components"].update({
        "securitySchemes": {
            "Bearer": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "UUID",
                "description": "Required format: Bearer <your-secret-key>"
            }
        }
    })
    
    openapi_schema.setdefault("security", []).append({"Bearer": []})
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

app.include_router(images.router)
app.include_router(chat.router)
app.include_router(models.router)
app.include_router(system.router)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port
    )