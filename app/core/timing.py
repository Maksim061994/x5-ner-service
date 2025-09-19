"""Утилиты для измерения времени выполнения и таймаутов."""

import asyncio
import time
import functools
from typing import Callable, Any, TypeVar
from app.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


def timeout_decorator(timeout_ms: int):
    """Декоратор для установки таймаута на асинхронные функции."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_ms / 1000.0
                )
            except asyncio.TimeoutError:
                logger.warning(f"Function {func.__name__} timed out after {timeout_ms}ms")
                raise
        return wrapper
    return decorator


class Timer:
    """Контекстный менеджер для измерения времени выполнения."""
    
    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = (self.end_time - self.start_time) * 1000  # в миллисекундах
        logger.info(f"{self.name} completed in {self.duration:.2f}ms")
    
    def get_duration_ms(self) -> float:
        """Получение времени выполнения в миллисекундах."""
        if self.duration is None:
            return (time.time() - self.start_time) * 1000
        return self.duration


def measure_time(func: Callable[..., T]) -> Callable[..., T]:
    """Декоратор для измерения времени выполнения функции."""
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> T:
        with Timer(func.__name__):
            return await func(*args, **kwargs)
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> T:
        with Timer(func.__name__):
            return func(*args, **kwargs)
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


class PerformanceMonitor:
    """Монитор производительности для отслеживания метрик."""
    
    def __init__(self):
        self.metrics = {}
    
    def record_metric(self, name: str, value: float, unit: str = "ms"):
        """Запись метрики."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append({"value": value, "unit": unit, "timestamp": time.time()})
    
    def get_avg_metric(self, name: str) -> float:
        """Получение среднего значения метрики."""
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        values = [m["value"] for m in self.metrics[name]]
        return sum(values) / len(values)
    
    def get_metrics_summary(self) -> dict:
        """Получение сводки по метрикам."""
        summary = {}
        for name, metrics in self.metrics.items():
            if metrics:
                values = [m["value"] for m in metrics]
                summary[name] = {
                    "count": len(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "unit": metrics[0]["unit"]
                }
        return summary


# Глобальный монитор производительности
performance_monitor = PerformanceMonitor()
