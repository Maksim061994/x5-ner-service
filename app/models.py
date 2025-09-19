"""Pydantic модели для запросов и ответов API."""

from typing import List, Literal
from pydantic import BaseModel, constr


class PredictRequest(BaseModel):
    """Модель запроса для предсказания сущностей."""
    input: str


class PredictSpan(BaseModel):
    """Модель спана сущности с BIO-разметкой."""
    start_index: int
    end_index: int
    entity: Literal[
        "B-TYPE", "I-TYPE",
        "B-BRAND", "I-BRAND", 
        "B-VOLUME", "I-VOLUME",
        "B-PERCENT", "I-PERCENT"
    ]


class HealthResponse(BaseModel):
    """Модель ответа для health check."""
    status: str


class ReadyResponse(BaseModel):
    """Модель ответа для readiness check."""
    model_config = {"protected_namespaces": ()}
    
    status: str
    pipeline_ready: bool
    model_loading_status: str
    model_name: str = None


class ModelStatusResponse(BaseModel):
    """Модель ответа для статуса модели."""
    model_config = {"protected_namespaces": ()}
    
    model_name: str = None
    loading_status: str  # "not_started", "loading", "loaded", "failed"
    loading_progress: float = 0.0  # 0.0 - 1.0
    error_message: str = None
