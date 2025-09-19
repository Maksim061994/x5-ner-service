"""Детектор брендов и типов товаров с поддержкой NER-моделей.

ВНИМАНИЕ: Этот файл создан как плейсхолдер согласно требованиям ТЗ.
В продакшене НЕ ИСПОЛЬЗУЙТЕ ручные словари брендов/типов!
Вместо этого используйте обученные NER-модели.
"""

from typing import List, Set, Optional, Any
import asyncio

# Импорты для работы с NER-моделью
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    print(f"Transformers library not available: {e}")
except Exception as e:
    TRANSFORMERS_AVAILABLE = False
    print(f"Error loading transformers library: {e}")


class BrandTypeDetector:
    """Детектор брендов и типов товаров с поддержкой NER-моделей."""
    
    def __init__(self):
        # ПУСТЫЕ словари - согласно требованиям ТЗ
        self.brands: Set[str] = set()
        self.types: Set[str] = set()
        self.ner_pipeline: Optional[Any] = None
        self._initialized = False
    
    async def detect_brands(self, text: str) -> List[str]:
        """Обнаружение брендов в тексте через NER-модель."""
        if self.ner_pipeline is None:
            return []
        
        try:
            # Выполняем предсказание в отдельном потоке
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, 
                lambda: self.ner_pipeline(text)
            )
            
            brands = []
            for entity in results:
                # Фильтруем только бренды
                if entity.get('entity_group') in ['BRAND', 'ORG', 'MISC'] or 'brand' in entity.get('entity_group', '').lower():
                    brands.append(entity['word'])
            
            return brands
            
        except Exception as e:
            print(f"Error detecting brands: {e}")
            return []
    
    async def detect_types(self, text: str) -> List[str]:
        """Обнаружение типов товаров в тексте через NER-модель."""
        if self.ner_pipeline is None:
            return []
        
        try:
            # Выполняем предсказание в отдельном потоке
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, 
                lambda: self.ner_pipeline(text)
            )
            
            types = []
            for entity in results:
                # Фильтруем только типы товаров
                if entity.get('entity_group') in ['TYPE', 'PRODUCT', 'MISC'] or 'type' in entity.get('entity_group', '').lower():
                    types.append(entity['word'])
            
            return types
            
        except Exception as e:
            print(f"Error detecting types: {e}")
            return []
    
    async def initialize_ner_model(self, model_name: str):
        """Инициализация NER-модели для детектора."""
        if not TRANSFORMERS_AVAILABLE:
            print("Transformers library not available")
            return
        
        try:
            print(f"Loading NER model for BrandTypeDetector: {model_name}")
            
            # Загружаем модель в отдельном потоке
            loop = asyncio.get_event_loop()
            self.ner_pipeline = await loop.run_in_executor(
                None, 
                lambda: pipeline(
                    "ner", 
                    model=model_name,
                    aggregation_strategy="simple",
                    device=-1  # CPU по умолчанию
                )
            )
            
            self._initialized = True
            print(f"NER model {model_name} loaded successfully for BrandTypeDetector")
            
        except Exception as e:
            print(f"Failed to load NER model {model_name} for BrandTypeDetector: {e}")
            self.ner_pipeline = None
    
    def is_initialized(self) -> bool:
        """Проверка инициализации детектора."""
        return self._initialized
