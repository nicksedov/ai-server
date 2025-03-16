from fastapi import APIRouter
import requests
import datetime

router = APIRouter(prefix="/v1")

@router.get("/health")
async def healthcheck():
    return {"status": "OK", "timestamp": datetime.datetime.now().isoformat()}