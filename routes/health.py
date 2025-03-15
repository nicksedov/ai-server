from fastapi import APIRouter
import requests

router = APIRouter(prefix="/v1")

@router.get("/health")
async def healthcheck():
    return {"status": "OK"}