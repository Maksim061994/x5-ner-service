import uuid
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import orjson

from app.api import router
from app.service.pipeline import Pipeline
from app.core.logging import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Starting Optimized X5 NER Service")
    
    try:
        pipeline = Pipeline()
        await pipeline.initialize()
        logger.info("Pipeline initialized successfully")
        app.state.pipeline = pipeline
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise
    yield
    logger.info("Shutting down X5 NER Service")


def create_optimized_app() -> FastAPI:
    app = FastAPI(
        title="Optimized X5 NER Service",
        description="Сервис извлечения сущностей",
        version="2.0.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["POST", "GET"],
        allow_headers=["*"],
    )
    
    @app.middleware("http")
    async def optimized_request_middleware(request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = (time.perf_counter() - start_time) * 1000
        
        if request.url.path == "/api/predict":
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.2f}"
        
        return response
    app.include_router(router)
    return app


class OptimizedJSONResponse(JSONResponse):

    def render(self, content) -> bytes:
        return orjson.dumps(
            content,
            option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_OMIT_MICROSECONDS
        )


app = create_optimized_app()
app.default_response_class = OptimizedJSONResponse
