"""Конфигурация приложения через переменные окружения."""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Настройки приложения."""
    
    # Основные настройки сервера
    PORT: int = 8000
    WORKERS: int = 1
    HOST: str = "0.0.0.0"
    
    # Настройки производительности
    TIMEOUT_MS: int = 900  # Таймаут для обработки запроса в миллисекундах
    
    # Настройки модели
    MODEL_NAME: Optional[str] = None  # Имя модели HuggingFace для NER
    
    # Настройки логирования
    LOG_LEVEL: str = "info"
    LOG_FORMAT: str = "json"  # json или text
    
    # Настройки кэширования
    CACHE_SIZE: int = 1000
    CACHE_TTL: int = 3600  # TTL кэша в секундах
    
    # Настройки rate limiting
    RATE_LIMIT_REQUESTS: int = 100  # Максимум запросов в минуту
    RATE_LIMIT_WINDOW: int = 60  # Окно для rate limiting в секундах
    
    # Настройки CORS
    CORS_ORIGINS: list = ["*"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Глобальный экземпляр настроек
settings = Settings()
