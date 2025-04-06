from fastapi import APIRouter, Depends, HTTPException
from schemas import SystemInfoResponse, HealthResponse, TorchInfoResponse, OllamaInfoResponse
from services.system_service import SystemService
from auth import verify_auth
import logging
import datetime

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)
system_service = SystemService()

@router.get("/system/info", 
           response_model=SystemInfoResponse,
           dependencies=[Depends(verify_auth)])
async def get_system_info():
    """Get detailed system health information"""
    try:
        system_info = system_service.get_system_info()
        return system_info
    except Exception as e:
        logger.error(f"Failed to get system info: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve system information"
        )

# Добавляем новый эндпоинт
@router.get("/torch/info", 
           response_model=TorchInfoResponse,
           dependencies=[Depends(verify_auth)])
async def get_torch_info():
    """Get detailed PyTorch diagnostic information"""
    try:
        return system_service.get_torch_info()
    except Exception as e:
        logger.error(f"Failed to get torch info: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve PyTorch information"
        )

# Добавляем новый эндпоинт
@router.get("/ollama/info", 
          response_model=OllamaInfoResponse,
          dependencies=[Depends(verify_auth)])
async def get_ollama_info():
    """Get Ollama service diagnostic information"""
    try:
        return system_service.get_ollama_info()
    except Exception as e:
        logger.error(f"Ollama info error: {str(e)}", exc_info=True)
        return OllamaInfoResponse(
            status="error",
            error=str(e)
        )

@router.get("/health",
           response_model=HealthResponse)
async def healthcheck():
    """Basic health check endpoint"""
    try:
        return {
            "status": "OK",
            "timestamp": datetime.datetime.now().isoformat(),
            "components": {
                "api": True,
                "models_cache": system_service.check_cache_health(),
                "gpu": system_service.check_gpu_health()
            }
        }
    except Exception as e:
        logger.error(f"Healthcheck failed: {str(e)}", exc_info=True)
        return {
            "status": "ERROR",
            "timestamp": datetime.datetime.now().isoformat(),
            "error": str(e)
        }