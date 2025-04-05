from fastapi import APIRouter, HTTPException, Depends, status
from schemas import DownloadRequest, DeleteRequest
from services.model_service import ModelService
from config import config
from models_cache import model_cache
from auth import verify_auth
import logging
import asyncio

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)
model_service = ModelService()

@router.get("/models", dependencies=[Depends(verify_auth)])
async def list_models():
    """Get list of available models from all providers"""
    try:
        models = await model_service.get_all_models()
        return format_models_response(models)
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to retrieve models list"
        )

@router.post("/models/download", 
            dependencies=[Depends(verify_auth)],
            status_code=status.HTTP_202_ACCEPTED)
async def download_model(request: DownloadRequest):
    """Download model from specified provider"""
    try:
        result = await model_service.download_model(
            request.provider,
            request.model_id,
            request.revision
        )
        await model_cache.refresh()
        return format_download_response(result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Model download failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model download failed: {str(e)}"
        )

@router.delete("/models", 
             dependencies=[Depends(verify_auth)],
             status_code=status.HTTP_202_ACCEPTED)
async def delete_model(request: DeleteRequest):
    """Delete model from specified provider"""
    try:
        result = await model_service.delete_model(
            request.provider,
            request.model_id,
            request.purge
        )
        await model_cache.refresh()
        return format_delete_response(result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Model deletion failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model deletion failed: {str(e)}"
        )

def format_models_response(models: list) -> dict:
    return {
        "object": "list",
        "data": sorted(
            [{
                "id": m["id"],
                "object": "model",
                "created": m["created"],
                "owned_by": m["owned_by"]
            } for m in models],
            key=lambda x: x["created"],
            reverse=True
        )
    }

def format_download_response(result: dict) -> dict:
    return {
        "status": result.get("status", "success"),
        "message": result.get("message", ""),
        "provider": result.get("provider", "unknown"),
        "model_id": result.get("model_id", ""),
        "details": result.get("details", {})
    }

def format_delete_response(result: dict) -> dict:
    return {
        "status": result.get("status", "success"),
        "message": result.get("message", ""),
        "provider": result.get("provider", "unknown"),
        "model_id": result.get("model_id", ""),
        "purged": result.get("purged", False)
    }