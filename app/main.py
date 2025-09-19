"""Основной модуль FastAPI приложения."""

import uuid
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import router
from app.service.pipeline import Pipeline
from app.core.config import settings
from app.core.logging import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения."""
    # Startup
    setup_logging()
    logger.info("Starting X5 NER Service")
    
    # Инициализация пайплайна
    try:
        pipeline = Pipeline()
        await pipeline.initialize()
        logger.info("Pipeline initialized successfully")
        # Сохраняем пайплайн в состоянии приложения
        app.state.pipeline = pipeline
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise
    
    yield
    # Shutdown
    logger.info("Shutting down X5 NER Service")


def create_app() -> FastAPI:
    """Создание и настройка FastAPI приложения."""
    app = FastAPI(
        title="X5 NER Service",
        description="Сервис извлечения сущностей из поисковых запросов Пятёрочки",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Middleware для генерации request_id
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
    
    # Подключение роутов
    app.include_router(router)
    
    return app


app = create_app()
