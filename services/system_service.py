# services/system_service.py
import psutil
import pynvml
import platform
import torch
import time
from typing import List, Dict
from schemas import CPUInfo, MemoryInfo, GPUInfo, SystemInfoResponse

class SystemService:
    def __init__(self):
        self.nvml_initialized = False
        self._initialize_nvml()

    def get_system_info(self) -> SystemInfoResponse:
        return SystemInfoResponse(
            timestamp=time.time(),
            os=self._get_os_info(),
            cpu=self._get_cpu_info(),
            memory=self._get_memory_info(),
            cuda_available=torch.cuda.is_available(),
            gpus=self._get_gpu_info(),
            torch_version=torch.__version__,
            python_version=platform.python_version()
        )

    def check_cache_health(self) -> bool:
        try:
            return psutil.virtual_memory().available > 512 * 1024 * 1024  # 512MB
        except:
            return False

    def check_gpu_health(self) -> Dict:
        if not torch.cuda.is_available():
            return {"available": False}
        
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            return {
                "available": True,
                "device_count": device_count,
                "memory_used_percent": self._get_gpu_memory_usage()
            }
        except:
            return {"available": False}
        finally:
            if self.nvml_initialized:
                pynvml.nvmlShutdown()

    def _initialize_nvml(self):
        try:
            pynvml.nvmlInit()
            self.nvml_initialized = True
        except:
            self.nvml_initialized = False

    def _get_os_info(self) -> str:
        return f"{platform.system()} {platform.release()}"

    def _get_cpu_info(self) -> CPUInfo:
        load = psutil.getloadavg()
        return CPUInfo(
            usage_percent=psutil.cpu_percent(),
            cores_physical=psutil.cpu_count(logical=False),
            cores_logical=psutil.cpu_count(logical=True),
            frequency_current=psutil.cpu_freq().current,
            frequency_max=psutil.cpu_freq().max,
            load_1m=load[0],
            load_5m=load[1],
            load_15m=load[2]
        )

    def _get_memory_info(self) -> MemoryInfo:
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return MemoryInfo(
            total=mem.total / (1024**3),
            available=mem.available / (1024**3),
            used=mem.used / (1024**3),
            percent=mem.percent,
            swap_total=swap.total / (1024**3),
            swap_used=swap.used / (1024**3)
        )

    def _get_gpu_info(self) -> List[GPUInfo]:
        if not self.nvml_initialized:
            return []
        
        try:
            return [self._get_gpu_details(i) for i in range(pynvml.nvmlDeviceGetCount())]
        except:
            return []
        finally:
            pynvml.nvmlShutdown()

    def _get_gpu_details(self, index: int) -> GPUInfo:
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        
        return GPUInfo(
            name=pynvml.nvmlDeviceGetName(handle),
            memory_total=memory.total // (1024**2),
            memory_used=memory.used // (1024**2),
            memory_free=memory.free // (1024**2),
            utilization=utilization.gpu,
            temperature=pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
            fan_speed_percent=self._get_fan_speed(handle)
        )

    def _get_fan_speed(self, handle) -> int:
        try:
            return pynvml.nvmlDeviceGetFanSpeed(handle)
        except:
            return None

    def _get_gpu_memory_usage(self) -> float:
        if not self.nvml_initialized:
            return 0.0
        
        try:
            pynvml.nvmlInit()
            total_used = 0
            total_total = 0
            
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_used += info.used
                total_total += info.total
                
            return (total_used / total_total) * 100 if total_total > 0 else 0.0
        except:
            return 0.0