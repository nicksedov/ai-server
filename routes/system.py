from fastapi import APIRouter, Depends
from typing import List
from schemas import CPUInfo,MemoryInfo,GPUInfo,SystemInfoResponse,HealthResponse
import psutil
import pynvml
import platform
import torch
import time
import datetime
from pydantic import BaseModel
from auth import verify_auth

router = APIRouter(prefix="/v1")

try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

def get_cpu_info() -> CPUInfo:
    load_avg = psutil.getloadavg()
    return CPUInfo(
        usage_percent=psutil.cpu_percent(),
        cores_physical=psutil.cpu_count(logical=False),
        cores_logical=psutil.cpu_count(logical=True),
        frequency_current=psutil.cpu_freq().current,
        frequency_max=psutil.cpu_freq().max,
        load_1m=load_avg[0] if load_avg else None,
        load_5m=load_avg[1] if load_avg else None,
        load_15m=load_avg[2] if load_avg else None
    )

def get_memory_info() -> MemoryInfo:
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return MemoryInfo(
        total=mem.total / (1024**3),
        available=mem.available / (1024**3),
        used=mem.used / (1024**3),
        percent=mem.percent,
        swap_total=swap.total / (1024**3) if swap.total > 0 else None,
        swap_used=swap.used / (1024**3) if swap.used > 0 else None
    )

def get_gpu_info() -> List[GPUInfo]:
    if not HAS_NVML or not torch.cuda.is_available():
        return []
    
    pynvml.nvmlInit()
    gpus = []
    
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # Получение информации о вентиляторах
            try:
                fan_speed_percent = pynvml.nvmlDeviceGetFanSpeed(handle)
            except pynvml.NVMLError:
                fan_speed_percent = None

            # Информация об имени устройства
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')  # Декодируем только байтовые строки
            
            # Обработка температуры с проверкой ошибок
            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except pynvml.NVMLError:
                temp = 0
                
            gpus.append(GPUInfo(
                name=name,
                memory_total=info.total // (1024**2),
                memory_used=info.used // (1024**2),
                memory_free=info.free // (1024**2),
                utilization=util.gpu,
                temperature=temp,
                fan_speed_percent=fan_speed_percent
            ))
            
    except Exception as e:
        logger.error(f"GPU info error: {str(e)}")
    finally:
        pynvml.nvmlShutdown()
    
    return gpus

@router.get("/system/info", 
           response_model=SystemInfoResponse,
           dependencies=[Depends(verify_auth)])
async def get_system_info():
    return {
        "timestamp": time.time(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu": get_cpu_info(),
        "memory": get_memory_info(),
        "cuda_available": torch.cuda.is_available(),
        "gpus": get_gpu_info(),
        "torch_version": torch.__version__,
        "python_version": platform.python_version()
    }

@router.get("/health",
           response_model=HealthResponse)
async def healthcheck():
    return {
        "status": "OK", 
        "timestamp": datetime.datetime.now().isoformat()
    }