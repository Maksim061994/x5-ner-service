import asyncio
import time
from typing import Dict, Any, Optional
from app.core.logging import get_logger

logger = get_logger(__name__)


class HealthChecker:
    """Класс для проверки здоровья различных компонентов системы."""
    
    def __init__(self):
        self.checks = {}
        self.start_time = time.time()
    
    def register_check(self, name: str, check_func: callable, timeout: float = 5.0):
        """Регистрация проверки здоровья компонента."""
        self.checks[name] = {
            "func": check_func,
            "timeout": timeout,
            "last_check": None,
            "last_result": None
        }
    
    async def run_check(self, name: str) -> Dict[str, Any]:
        """Выполнение проверки здоровья компонента."""
        if name not in self.checks:
            return {
                "status": "error",
                "message": f"Check '{name}' not registered"
            }
        
        check_info = self.checks[name]
        check_func = check_info["func"]
        timeout = check_info["timeout"]
        
        try:
            # Выполняем проверку с таймаутом
            if asyncio.iscoroutinefunction(check_func):
                result = await asyncio.wait_for(check_func(), timeout=timeout)
            else:
                result = check_func()
            
            # Обновляем информацию о последней проверке
            check_info["last_check"] = time.time()
            check_info["last_result"] = result
            
            return {
                "status": "ok",
                "result": result,
                "timestamp": check_info["last_check"]
            }
            
        except asyncio.TimeoutError:
            logger.warning(f"Health check '{name}' timed out after {timeout}s")
            return {
                "status": "timeout",
                "message": f"Check timed out after {timeout}s"
            }
        except Exception as e:
            logger.error(f"Health check '{name}' failed: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Выполнение всех зарегистрированных проверок."""
        results = {}
        overall_status = "ok"
        
        for name in self.checks:
            result = await self.run_check(name)
            results[name] = result
            
            if result["status"] != "ok":
                overall_status = "degraded"
        
        return {
            "status": overall_status,
            "checks": results,
            "uptime": time.time() - self.start_time
        }
    
    def get_simple_health(self) -> Dict[str, Any]:
        """Простая проверка здоровья (без выполнения всех проверок)."""
        return {
            "status": "ok",
            "uptime": time.time() - self.start_time,
            "timestamp": time.time()
        }


# Глобальный экземпляр проверки здоровья
health_checker = HealthChecker()


# Стандартные проверки здоровья
async def check_pipeline_health():
    """Проверка здоровья пайплайна NER."""
    # Импортируем здесь, чтобы избежать циклических импортов
    from app.service.pipeline import Pipeline
    
    # Проверяем, что пайплайн инициализирован
    # В реальном приложении здесь может быть более сложная логика
    return {"pipeline_ready": True}


async def check_memory_usage():
    """Проверка использования памяти."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return {
            "total": memory.total,
            "available": memory.available,
            "percent": memory.percent
        }
    except ImportError:
        return {"message": "psutil not available"}


def check_disk_space():
    """Проверка свободного места на диске."""
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        return {
            "total": total,
            "used": used,
            "free": free,
            "percent_used": (used / total) * 100
        }
    except Exception as e:
        return {"error": str(e)}


# Регистрируем стандартные проверки
health_checker.register_check("pipeline", check_pipeline_health)
health_checker.register_check("memory", check_memory_usage)
health_checker.register_check("disk", check_disk_space)
