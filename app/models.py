from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class PredictRequest(BaseModel):
    """модель запроса для предсказания сущностей."""
    input: str = Field(..., min_length=1, max_length=10000)
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        validate_default=False,
        extra="forbid"
    )


class PredictSpan(BaseModel):
    """модель спана сущности с BIO-разметкой."""
    start_index: int = Field(..., ge=0)
    end_index: int = Field(..., ge=0)
    entity: str = Field(..., min_length=1, max_length=20)
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )


class HealthResponse(BaseModel):
    """модель ответа для health check."""
    status: str = Field(default="ok", min_length=1, max_length=10)
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid"
    )


class ReadyResponse(BaseModel):
    """модель ответа для readiness check."""
    status: str = Field(..., min_length=1, max_length=20)
    pipeline_ready: bool = Field(...)
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid"
    )


class ModelStatusResponse(BaseModel):
    """модель ответа для статуса модели."""
    loading_status: str = Field(..., min_length=1, max_length=20)
    model_name: Optional[str] = Field(default=None, max_length=100)
    loading_progress: float = Field(default=0.0, ge=0.0, le=1.0)
    error_message: Optional[str] = Field(default=None, max_length=500)
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid"
    )


# Быстрые функции для работы с данными
def serialize_spans_fast(spans: List[tuple]) -> List[dict]:
    """сериализация спанов без создания объектов."""
    return [
        {
            "start_index": span[0],
            "end_index": span[1], 
            "entity": validate_entity(span[2])
        }
        for span in spans
    ]


def validate_input_fast(text: str) -> str:
    """валидация входного текста."""
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    
    text = text.strip()
    if not text:
        raise ValueError("Input cannot be empty")
    
    if len(text) > 10000:
        raise ValueError("Input too long")
    
    return text


# Валидация entity типов
VALID_ENTITIES = {
    "B-TYPE", "I-TYPE", "B-BRAND", "I-BRAND", 
    "B-VOLUME", "I-VOLUME", "B-PERCENT", "I-PERCENT", "O"
}

def validate_entity(entity: str) -> str:
    """валидация entity типа."""
    if entity not in VALID_ENTITIES:
        return "O"  # По умолчанию
    return entity


# Оптимизированные методы для Pydantic моделей
class OptimizedPredictRequest(PredictRequest):
    """Оптимизированная версия PredictRequest с быстрой валидацией."""
    
    @classmethod
    def from_dict(cls, data: dict) -> "OptimizedPredictRequest":
        """Быстрое создание из словаря."""
        return cls(input=data["input"])
    
    def to_dict(self) -> dict:
        """Быстрое преобразование в словарь."""
        return {"input": self.input}


class OptimizedPredictSpan(PredictSpan):
    """Оптимизированная версия PredictSpan с быстрой валидацией."""
    
    @classmethod
    def from_tuple(cls, span: tuple) -> "OptimizedPredictSpan":
        """Быстрое создание из кортежа."""
        return cls(
            start_index=span[0],
            end_index=span[1],
            entity=validate_entity(span[2])
        )
    
    def to_dict(self) -> dict:
        """Быстрое преобразование в словарь."""
        return {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "entity": self.entity
        }


# Функции для совместимости с FastAPI
def get_response_model(model_class):
    """Получение модели ответа для FastAPI."""
    return model_class


def get_request_model(model_class):
    """Получение модели запроса для FastAPI."""
    return model_class


# Быстрые функции создания объектов
def create_predict_request(input_text: str) -> PredictRequest:
    """Быстрое создание PredictRequest."""
    return PredictRequest(input=input_text)


def create_predict_span(start: int, end: int, entity: str) -> PredictSpan:
    """Быстрое создание PredictSpan."""
    return PredictSpan(start_index=start, end_index=end, entity=entity)


def create_health_response(status: str = "ok") -> HealthResponse:
    """Быстрое создание HealthResponse."""
    return HealthResponse(status=status)


def create_ready_response(status: str, pipeline_ready: bool) -> ReadyResponse:
    """Быстрое создание ReadyResponse."""
    return ReadyResponse(status=status, pipeline_ready=pipeline_ready)


def create_model_status_response(
    loading_status: str,
    model_name: Optional[str] = None,
    loading_progress: float = 0.0,
    error_message: Optional[str] = None
) -> ModelStatusResponse:
    """Быстрое создание ModelStatusResponse."""
    return ModelStatusResponse(
        loading_status=loading_status,
        model_name=model_name,
        loading_progress=loading_progress,
        error_message=error_message
    )
