from typing import Optional
import spacy
from app.service.features import predict_spans_spacy
from app.service.postprocess import spans_to_bio_splits, convert_pred_to_output
from app.core.logging import get_logger
from app.core.config import settings
import sys
from app.service.stacked_crf import StackedCRF
from app.service.features import FeatureBuilder, FeatureConfig
import joblib

sys.modules['__main__'].StackedCRF = StackedCRF
sys.modules['__main__'].FeatureBuilder = FeatureBuilder
sys.modules['__main__'].FeatureConfig = FeatureConfig


logger = get_logger(__name__)


class Pipeline:
    
    def __init__(self):
        self._initialized = False
        
        # Статус загрузки модели
        self._model_spacy_best: Optional[str] = None
        self._model_stack_crf: Optional[str] = None
        self._model_spacy_full: Optional[str] = None
        self._model_loading_status: str = "not_started"
        self._models_path: str = settings.MODELS_PATH

    def load_crf_model(self):
        model = joblib.load(f"{self._models_path}/crf/StackedCRF.joblib")
        return model

    def load_spacy_model(self, name_model: str):
        model = spacy.load(f"{self._models_path}/{name_model}")
        return model

    async def initialize(self):
        """Асинхронная инициализация компонентов пайплайна."""
        try:
            logger.info("Initializing NER pipeline components...")
            
            # Инициализируем компоненты
            self._model_stack_crf = self.load_crf_model()
            self._model_spacy_best = self.load_spacy_model(name_model="spacy_best")
            self._model_spacy_full = self.load_spacy_model(name_model="spacy_full")
            
            self._initialized = True
            logger.info("NER pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Проверка готовности пайплайна."""
        return self._initialized and all([
            self._model_stack_crf is not None,
            self._model_spacy_best is not None,
            self._model_spacy_full is not None
        ])

    def _predict_crf_pipline(self, text):
        spans = self._model_stack_crf.predict_spans([text])
        predict = spans_to_bio_splits(text, spans[0])
        return predict

    def _predict_spacy(self, text, name_model="best"):
        if name_model == "best":
            model = self._model_spacy_best
        else:
            model = self._model_spacy_full
        spans = predict_spans_spacy(model, text)
        predict = convert_pred_to_output(text, spans)
        return predict

    async def predict_result(self, text: str):
        if not self.is_ready():
            raise RuntimeError("Pipeline is not ready")

        if not text or not text.strip():
            return []

        predict_crf = self._predict_crf_pipline(text)
        predict_spacy_best = self._predict_spacy(text, name_model="best")

        result = predict_crf
        if predict_crf != predict_spacy_best and len(text) < 10: # spacy дает более уверенный результат
            predict_spacy_full = self._predict_spacy(text, name_model="full")

            if predict_spacy_full == predict_spacy_best:
                result = predict_spacy_best

        return result

    async def predict_bio(self, text: str):
        """Основной метод предсказания"""
        return await self.predict_result(text)

    def get_model_name(self) -> str:
        """Получение имени модели."""
        return "NER Pipeline"

    def get_model_loading_status(self) -> str:
        """Получение статуса загрузки модели."""
        return self._model_loading_status
