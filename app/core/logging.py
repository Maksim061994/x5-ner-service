"""Настройка логирования для приложения."""

import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict
from app.core.config import settings


class JSONFormatter(logging.Formatter):
    """JSON форматтер для логов."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Форматирование лога в JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Добавляем дополнительные поля если есть
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        
        if hasattr(record, 'processing_time'):
            log_entry['processing_time_ms'] = record.processing_time
        
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Текстовый форматтер для логов."""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


def setup_logging():
    """Настройка системы логирования."""
    # Уровень логирования
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    # Создаем root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Удаляем существующие обработчики
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Создаем обработчик для stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    
    # Выбираем форматтер
    if settings.LOG_FORMAT.lower() == "json":
        formatter = JSONFormatter()
    else:
        formatter = TextFormatter()
    
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    
    # Настраиваем логирование для внешних библиотек
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Получение логгера с указанным именем."""
    return logging.getLogger(name)


def log_request(logger: logging.Logger, request_id: str, message: str, **kwargs):
    """Логирование запроса с request_id."""
    extra = {"request_id": request_id}
    extra.update(kwargs)
    logger.info(message, extra=extra)


def log_performance(logger: logging.Logger, request_id: str, processing_time: float, **kwargs):
    """Логирование производительности запроса."""
    extra = {
        "request_id": request_id,
        "processing_time": processing_time
    }
    extra.update(kwargs)
    logger.info(f"Request processed in {processing_time:.2f}ms", extra=extra)
