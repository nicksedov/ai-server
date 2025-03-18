from fastapi import FastAPI
from routes import images, chat, models, health
from config import config
import uvicorn

app = FastAPI()

app.include_router(images.router)
app.include_router(chat.router)
app.include_router(models.router)
app.include_router(health.router)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port
    )