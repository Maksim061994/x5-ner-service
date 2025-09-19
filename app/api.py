"""API роуты для FastAPI приложения."""

import asyncio
import time
from typing import List
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.models import PredictRequest, PredictSpan, HealthResponse, ReadyResponse, ModelStatusResponse
from app.service.pipeline import Pipeline
from app.core.config import settings
from app.core.logging import get_logger
from app.core.timing import timeout_decorator

logger = get_logger(__name__)
router = APIRouter()

# Pipeline initialization moved to main.py lifespan manager


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка здоровья сервиса."""
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse)
async def readiness_check(request: Request):
    """Проверка готовности сервиса к обработке запросов."""
    pipeline = getattr(request.app.state, 'pipeline', None)
    pipeline_ready = pipeline is not None and pipeline.is_ready()
    
    # Получаем статус загрузки модели
    model_status = "not_started"
    model_name = None
    if pipeline:
        model_status = pipeline.get_model_loading_status()
        model_name = pipeline.get_model_name()
    
    return ReadyResponse(
        status="ok" if pipeline_ready else "not ready",
        pipeline_ready=pipeline_ready,
        model_loading_status=model_status,
        model_name=model_name
    )


@router.get("/model/status", response_model=ModelStatusResponse)
async def model_status_check(request: Request):
    """Проверка статуса загрузки модели."""
    pipeline = getattr(request.app.state, 'pipeline', None)
    
    if not pipeline:
        return ModelStatusResponse(
            loading_status="not_started",
            loading_progress=0.0,
            error_message="Pipeline not initialized"
        )
    
    return ModelStatusResponse(
        model_name=pipeline.get_model_name(),
        loading_status=pipeline.get_model_loading_status(),
        loading_progress=pipeline.get_model_loading_progress(),
        error_message=pipeline.get_model_loading_error()
    )


@router.post("/api/predict", response_model=List[PredictSpan])
async def predict_entities(request: PredictRequest, http_request: Request):
    """Извлечение сущностей из текста с BIO-разметкой."""
    start_time = time.time()
    request_id = getattr(http_request.state, 'request_id', 'unknown')
    
    logger.info(f"Processing request {request_id}: input='{request.input[:50]}...'")
    
    # Если пустой ввод - возвращаем пустой список
    if not request.input:
        logger.info(f"Empty input for request {request_id}")
        return []
    
    pipeline = getattr(http_request.app.state, 'pipeline', None)
    if not pipeline or not pipeline.is_ready():
        logger.error(f"Pipeline not ready for request {request_id}")
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        # Применяем таймаут к предсказанию
        @timeout_decorator(timeout_ms=settings.TIMEOUT_MS)
        async def _predict_with_timeout():
            return await pipeline.predict_bio(request.input)
        
        spans = await _predict_with_timeout()
        
        # Преобразуем в формат ответа
        result = [
            PredictSpan(
                start_index=span.start,
                end_index=span.end,
                entity=span.entity
            )
            for span in spans
        ]
        
        processing_time = (time.time() - start_time) * 1000
        logger.info(
            f"Request {request_id} completed in {processing_time:.2f}ms, "
            f"found {len(result)} entities"
        )
        
        return result
        
    except asyncio.TimeoutError:
        processing_time = (time.time() - start_time) * 1000
        logger.warning(
            f"Request {request_id} timed out after {processing_time:.2f}ms, "
            f"returning empty result"
        )
        return []
        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        logger.error(
            f"Error processing request {request_id} after {processing_time:.2f}ms: {e}",
            exc_info=True
        )
        return []
