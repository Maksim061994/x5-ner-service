"""Конфигурация приложения через переменные окружения."""

from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Настройки приложения."""
    
    # Основные настройки сервера
    PORT: int = 8000
    HOST: str = "0.0.0.0"
     
    # Настройки модели
    MODEL_NAME: Optional[str] = None
    CUDA_DEVICE: int = -1
    
    # Настройки логирования
    LOG_LEVEL: str = "info"
    LOG_FORMAT: str = "json"  # json или text
     
    # Настройки CORS
    CORS_ORIGINS: list = ["*"]

    MODELS_PATH: str = "models"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


# Глобальный экземпляр настроек
settings = Settings()
