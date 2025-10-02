import asyncio
import time
from typing import List
from fastapi import APIRouter, HTTPException, Request

from app.models import (
    PredictRequest, PredictSpan, HealthResponse, ReadyResponse, ModelStatusResponse,
    serialize_spans_fast, validate_input_fast
)
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Оптимизированная проверка здоровья сервиса."""
    from app.models import create_health_response
    return create_health_response()


@router.get("/ready", response_model=ReadyResponse)
async def readiness_check(request: Request):
    """Оптимизированная проверка готовности сервиса."""
    from app.models import create_ready_response
    pipeline = getattr(request.app.state, 'pipeline', None)
    pipeline_ready = pipeline is not None and pipeline.is_ready()
    
    return create_ready_response(
        "ok" if pipeline_ready else "not ready",
        pipeline_ready
    )


@router.get("/model/status", response_model=ModelStatusResponse)
async def model_status_check(request: Request):
    """Оптимизированная проверка статуса модели."""
    from app.models import create_model_status_response
    pipeline = getattr(request.app.state, 'pipeline', None)
    
    if not pipeline:
        return create_model_status_response(
            loading_status="not_started",
            error_message="Pipeline not initialized"
        )
    
    return create_model_status_response(
        loading_status=pipeline.get_model_loading_status(),
        model_name=pipeline.get_model_name(),
        loading_progress=pipeline.get_model_loading_progress(),
        error_message=pipeline.get_model_loading_error()
    )


@router.post("/api/predict", response_model=List[PredictSpan])
async def predict_entities_optimized(request: PredictRequest, http_request: Request):
    """Оптимизированное извлечение сущностей с минимальными накладными расходами."""
    start_time = time.perf_counter()
    request_id = getattr(http_request.state, 'request_id', 'unknown')
    
    # Быстрая валидация входных данных
    try:
        text = validate_input_fast(request.input)
    except ValueError as e:
        logger.warning(f"Invalid input for request {request_id}: {e}")
        return []
    
    # Логирование только для длинных запросов
    if len(text) > 100:
        logger.info(f"Processing request {request_id}: input='{text[:50]}...'")
    
    # Если пустой ввод - быстрый возврат
    if not text:
        return []
    
    pipeline = getattr(http_request.app.state, 'pipeline', None)
    if not pipeline or not pipeline.is_ready():
        logger.error(f"Pipeline not ready for request {request_id}")
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        # Используем оптимизированный пайплайн
        spans = await pipeline.predict_bio(text)
        
        # Быстрая сериализация без создания объектов
        result = serialize_spans_fast(spans)
        
        # Преобразуем в PredictSpan объекты
        predict_spans = [
            PredictSpan(
                start_index=span["start_index"],
                end_index=span["end_index"],
                entity=span["entity"]
            )
            for span in result
        ]
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        # Логирование только для медленных запросов
        if processing_time > 500:
            logger.info(
                f"Request {request_id} completed in {processing_time:.2f}ms, "
                f"found {len(predict_spans)} entities"
            )
        
        return predict_spans
        
    except asyncio.TimeoutError:
        processing_time = (time.perf_counter() - start_time) * 1000
        logger.warning(
            f"Request {request_id} timed out after {processing_time:.2f}ms"
        )
        return []
        
    except Exception as e:
        processing_time = (time.perf_counter() - start_time) * 1000
        logger.error(
            f"Error processing request {request_id} after {processing_time:.2f}ms: {e}"
        )
        return []
