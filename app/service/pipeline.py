"""Основной пайплайн для извлечения сущностей с BIO-разметкой."""

import asyncio
from typing import List, Optional, Dict, Any
from functools import lru_cache

from app.service.rules import RegexRules, Span
from app.service.tokenizer import SimpleTokenizer
from app.service.postprocess import PostProcessor, ProcessedSpan
from app.service.brands_types import BrandTypeDetector
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Импорты для работы с NER-моделью
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers library loaded successfully")
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(f"Transformers library not available: {e}. NER model functionality will be disabled.")
except Exception as e:
    TRANSFORMERS_AVAILABLE = False
    logger.error(f"Error loading transformers library: {e}. NER model functionality will be disabled.")


class Pipeline:
    """Основной пайплайн для обработки NER задач."""
    
    def __init__(self):
        self.rules: Optional[RegexRules] = None
        self.tokenizer: Optional[SimpleTokenizer] = None
        self.postprocessor: Optional[PostProcessor] = None
        self.brand_type_detector: Optional[BrandTypeDetector] = None
        self.ner_pipeline: Optional[Any] = None
        self._initialized = False
        
        # Статус загрузки модели
        self._model_name: Optional[str] = None
        self._model_loading_status: str = "not_started"  # "not_started", "loading", "loaded", "failed"
        self._model_loading_progress: float = 0.0
        self._model_loading_error: Optional[str] = None
    
    async def initialize(self):
        """Асинхронная инициализация компонентов пайплайна."""
        try:
            logger.info("Initializing NER pipeline components...")
            
            # Инициализируем компоненты
            self.rules = RegexRules()
            self.tokenizer = SimpleTokenizer()
            self.postprocessor = PostProcessor()
            self.brand_type_detector = BrandTypeDetector()
            
            # Инициализация NER-модели
            if settings.MODEL_NAME and TRANSFORMERS_AVAILABLE:
                self._model_name = settings.MODEL_NAME
                await self._load_ner_model(settings.MODEL_NAME)
                # Также инициализируем NER-модель в BrandTypeDetector
                await self.brand_type_detector.initialize_ner_model(settings.MODEL_NAME)
            elif settings.MODEL_NAME and not TRANSFORMERS_AVAILABLE:
                logger.warning(f"Model {settings.MODEL_NAME} specified but transformers not available")
                self._model_name = settings.MODEL_NAME
                self._model_loading_status = "failed"
                self._model_loading_error = "Transformers library not available"
            else:
                logger.info("No NER model specified, using rule-based extraction only")
                self._model_loading_status = "not_started"
            
            self._initialized = True
            logger.info("NER pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Проверка готовности пайплайна."""
        return self._initialized and all([
            self.rules is not None,
            self.tokenizer is not None,
            self.postprocessor is not None,
            self.brand_type_detector is not None
        ])
    
    def get_model_name(self) -> Optional[str]:
        """Получение имени модели."""
        return self._model_name
    
    def get_model_loading_status(self) -> str:
        """Получение статуса загрузки модели."""
        return self._model_loading_status
    
    def get_model_loading_progress(self) -> float:
        """Получение прогресса загрузки модели (0.0 - 1.0)."""
        return self._model_loading_progress
    
    def get_model_loading_error(self) -> Optional[str]:
        """Получение ошибки загрузки модели."""
        return self._model_loading_error
    
    async def predict_bio(self, text: str) -> List[ProcessedSpan]:
        """Предсказание сущностей с BIO-разметкой."""
        if not self.is_ready():
            raise RuntimeError("Pipeline is not ready")
        
        if not text or not text.strip():
            return []
        
        # Нормализация текста
        normalized_text = self._normalize_text(text)
        
        # Извлечение сущностей
        spans = []
        
        # 1. Извлечение VOLUME и PERCENT через регулярные выражения
        rule_spans = self.rules.extract_all(normalized_text)
        spans.extend(rule_spans)
        
        # 2. Извлечение TYPE и BRAND через NER-модель
        if self.ner_pipeline is not None:
            brand_spans = await self._extract_brands(normalized_text)
            type_spans = await self._extract_types(normalized_text)
            logger.debug(f"NER extracted {len(brand_spans)} brands and {len(type_spans)} types")
            spans.extend(brand_spans)
            spans.extend(type_spans)
        else:
            # Fallback к базовому детектору, если NER-модель недоступна
            logger.debug("NER model not available, using fallback brand/type detection")
        
        # 3. Постобработка: устранение конфликтов и валидация BIO
        processed_spans = self.postprocessor.process_spans(spans)
        
        return processed_spans
    
    def _normalize_text(self, text: str) -> str:
        """Нормализация текста для обработки."""
        # Базовая нормализация: убираем лишние пробелы
        return ' '.join(text.split())
    
    @lru_cache(maxsize=1000)
    def _cached_predict(self, text: str) -> List[ProcessedSpan]:
        """Кэшированное предсказание для коротких строк."""
        # Синхронная версия для кэширования
        if not text or not text.strip():
            return []
        
        normalized_text = self._normalize_text(text)
        spans = self.rules.extract_all(normalized_text)
        processed_spans = self.postprocessor.process_spans(spans)
        return processed_spans
    
    async def _load_ner_model(self, model_name: str):
        """Загрузка NER-модели через HuggingFace Transformers."""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers library not available")
            self._model_loading_status = "failed"
            self._model_loading_error = "Transformers library not available"
            return
        
        try:
            logger.info(f"Loading NER model: {model_name}")
            self._model_loading_status = "loading"
            self._model_loading_progress = 0.1
            
            # Загружаем модель в отдельном потоке, чтобы не блокировать event loop
            loop = asyncio.get_event_loop()
            
            def load_model():
                self._model_loading_progress = 0.3
                return pipeline(
                    "ner", 
                    model=model_name,
                    aggregation_strategy="max",  # Для группировки токенов в сущности
                    device=-1  # CPU по умолчанию, можно изменить на GPU если доступен
                )
            
            self._model_loading_progress = 0.5
            self.ner_pipeline = await loop.run_in_executor(None, load_model)
            self._model_loading_progress = 0.9
            
            self._model_loading_status = "loaded"
            self._model_loading_progress = 1.0
            self._model_loading_error = None
            
            logger.info(f"NER model {model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load NER model {model_name}: {e}")
            self.ner_pipeline = None
            self._model_loading_status = "failed"
            self._model_loading_error = str(e)
            self._model_loading_progress = 0.0
    
    async def _extract_brands(self, text: str) -> List[Span]:
        """Извлечение брендов через NER-модель."""
        if self.ner_pipeline is None:
            return []
        
        # Пропускаем очень короткие тексты
        if len(text.strip()) < 3:
            return []
        
        try:
            # Выполняем предсказание в отдельном потоке
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, 
                lambda: self.ner_pipeline(text)
            )
            
            spans = []
            for entity in results:
                # Для bert-base-multilingual-cased используем LABEL_1 как потенциальные бренды
                entity_text = entity.get('word', '').strip()
                entity_group = entity.get('entity_group', '')
                score = entity.get('score', 0.0)
                
                # Критерии для определения бренда:
                # 1. LABEL_1 с высокой уверенностью
                # 2. Короткие фразы (1-3 слова)
                # 3. Содержат заглавные буквы или известные бренды
                if (entity_group == 'LABEL_1' and 
                    score > 0.4 and 
                    len(entity_text.split()) <= 3 and
                    self._looks_like_brand(entity_text)):
                    
                    span = Span(
                        start=entity['start'],
                        end=entity['end'],
                        text=entity_text,
                        entity='BRAND'
                    )
                    spans.append(span)
            
            return spans
            
        except Exception as e:
            logger.error(f"Error extracting brands: {e}")
            return []
    
    async def _extract_types(self, text: str) -> List[Span]:
        """Извлечение типов товаров через NER-модель."""
        if self.ner_pipeline is None:
            return []
        
        # Пропускаем очень короткие тексты
        if len(text.strip()) < 3:
            return []
        
        try:
            # Выполняем предсказание в отдельном потоке
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, 
                lambda: self.ner_pipeline(text)
            )
            
            spans = []
            for entity in results:
                # Для bert-base-multilingual-cased используем LABEL_0 как потенциальные типы товаров
                entity_text = entity.get('word', '').strip()
                entity_group = entity.get('entity_group', '')
                score = entity.get('score', 0.0)
                
                # Критерии для определения типа товара:
                # 1. LABEL_0 с высокой уверенностью
                # 2. Содержит слова, похожие на типы товаров
                # 3. Не является числом или единицей измерения
                if (entity_group == 'LABEL_0' and 
                    score > 0.4 and 
                    self._looks_like_product_type(entity_text)):
                    
                    span = Span(
                        start=entity['start'],
                        end=entity['end'],
                        text=entity_text,
                        entity='TYPE'
                    )
                    spans.append(span)
            
            return spans
            
        except Exception as e:
            logger.error(f"Error extracting types: {e}")
            return []
    
    def _looks_like_brand(self, text: str) -> bool:
        """Проверка, похож ли текст на название бренда."""
        if not text or len(text.strip()) < 2:
            return False
        
        text = text.strip()
        
        # Исключаем числа и единицы измерения
        if text.isdigit() or text in ['г', 'кг', 'л', 'мл', 'шт', '%']:
            return False
        
        # Исключаем очень длинные фразы
        if len(text.split()) > 3:
            return False
        
        # Проверяем наличие заглавных букв (характерно для брендов)
        has_uppercase = any(c.isupper() for c in text)
        
        # Проверяем известные паттерны брендов
        brand_indicators = [
            # Английские бренды
            'coca', 'cola', 'nike', 'adidas', 'apple', 'samsung', 'sony', 'lg',
            'microsoft', 'google', 'amazon', 'tesla', 'bmw', 'mercedes', 'audi',
            # Русские бренды
            'пятерочка', 'магнит', 'лента', 'ашан', 'перекресток', 'дикси',
            'активия', 'домик', 'простоквашино', 'веселый', 'молочник'
        ]
        
        text_lower = text.lower()
        is_known_brand = any(brand in text_lower for brand in brand_indicators)
        
        # Бренд если: известный бренд ИЛИ (заглавные буквы И не число)
        return is_known_brand or (has_uppercase and not text.isdigit())
    
    def _looks_like_product_type(self, text: str) -> bool:
        """Проверка, похож ли текст на тип товара."""
        if not text or len(text.strip()) < 2:
            return False
        
        text = text.strip()
        
        # Исключаем числа и единицы измерения
        if text.isdigit() or text in ['г', 'кг', 'л', 'мл', 'шт', '%']:
            return False
        
        # Исключаем очень длинные фразы
        if len(text.split()) > 4:
            return False
        
        # Проверяем, не содержит ли текст числа
        if any(char.isdigit() for char in text):
            return False
        
        # Проверяем известные типы товаров
        product_types = [
            # Молочные продукты
            'молоко', 'сыр', 'творог', 'йогурт', 'кефир', 'сметана', 'масло',
            'сосиска', 'колбаса', 'ветчина', 'бекон',
            # Хлебобулочные
            'хлеб', 'булка', 'батон', 'круассан', 'печенье', 'торт', 'пирог',
            # Напитки
            'сок', 'вода', 'чай', 'кофе', 'лимонад', 'квас',
            # Мясо и рыба
            'мясо', 'курица', 'говядина', 'свинина', 'рыба', 'креветки',
            # Овощи и фрукты
            'яблоко', 'банан', 'апельсин', 'помидор', 'огурец', 'картофель',
            # Сладости
            'конфеты', 'шоколад', 'мороженое', 'вафли', 'зефир',
            # Английские типы
            'bottle', 'can', 'box', 'bag', 'pack', 'bottle', 'shoes', 'shirt',
            'pants', 'dress', 'jacket', 'hat', 'gloves', 'socks'
        ]
        
        text_lower = text.lower()
        
        # Проверяем, содержит ли текст слова, похожие на типы товаров
        for product_type in product_types:
            if product_type in text_lower:
                return True
        
        # Дополнительные эвристики для типов товаров
        # Слова, заканчивающиеся на характерные суффиксы
        type_suffixes = ['ка', 'ок', 'ик', 'ец', 'ица', 'ище', 'ище', 'ище']
        for suffix in type_suffixes:
            if text_lower.endswith(suffix) and len(text) > 3:
                return True
        
        return False
