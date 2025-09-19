"""Постобработка спанов: устранение пересечений и валидация BIO-последовательности."""

from typing import List, Set
from dataclasses import dataclass
from app.service.rules import Span


@dataclass
class ProcessedSpan:
    """Обработанный спан с валидной BIO-разметкой."""
    start: int
    end: int
    entity: str
    text: str


class PostProcessor:
    """Класс для постобработки спанов сущностей."""
    
    # Приоритеты сущностей при конфликте (больше = выше приоритет)
    ENTITY_PRIORITIES = {
        "PERCENT": 4,
        "VOLUME": 3,
        "BRAND": 2,
        "TYPE": 1
    }
    
    def __init__(self):
        pass
    
    def resolve_conflicts(self, spans: List[Span]) -> List[Span]:
        """Устранение пересечений между спанами."""
        if not spans:
            return []
        
        # Сортируем спаны по начальной позиции
        sorted_spans = sorted(spans, key=lambda x: x.start)
        resolved_spans = []
        
        for current_span in sorted_spans:
            # Проверяем пересечения с уже добавленными спанами
            has_conflict = False
            
            for existing_span in resolved_spans:
                if self._spans_overlap(current_span, existing_span):
                    # Определяем приоритет
                    current_priority = self._get_entity_priority(current_span.entity)
                    existing_priority = self._get_entity_priority(existing_span.entity)
                    
                    if current_priority > existing_priority:
                        # Заменяем существующий спан
                        resolved_spans.remove(existing_span)
                        resolved_spans.append(current_span)
                    has_conflict = True
                    break
            
            if not has_conflict:
                resolved_spans.append(current_span)
        
        return resolved_spans
    
    def _spans_overlap(self, span1: Span, span2: Span) -> bool:
        """Проверка пересечения двух спанов."""
        return not (span1.end <= span2.start or span2.end <= span1.start)
    
    def _get_entity_priority(self, entity: str) -> int:
        """Получение приоритета сущности."""
        # Извлекаем тип сущности (убираем B-/I- префикс)
        entity_type = entity.split('-', 1)[1] if '-' in entity else entity
        return self.ENTITY_PRIORITIES.get(entity_type, 0)
    
    def validate_bio_sequence(self, spans: List[Span]) -> List[ProcessedSpan]:
        """Валидация и исправление BIO-последовательности."""
        if not spans:
            return []
        
        # Группируем спаны по типу сущности
        entity_groups = {}
        for span in spans:
            entity_type = span.entity.split('-', 1)[1] if '-' in span.entity else span.entity
            if entity_type not in entity_groups:
                entity_groups[entity_type] = []
            entity_groups[entity_type].append(span)
        
        processed_spans = []
        
        for entity_type, entity_spans in entity_groups.items():
            # Сортируем спаны по позиции
            sorted_spans = sorted(entity_spans, key=lambda x: x.start)
            
            for i, span in enumerate(sorted_spans):
                # Определяем BIO-тег
                if i == 0 or not self._spans_adjacent(sorted_spans[i-1], span):
                    bio_tag = f"B-{entity_type}"
                else:
                    bio_tag = f"I-{entity_type}"
                
                processed_spans.append(ProcessedSpan(
                    start=span.start,
                    end=span.end,
                    entity=bio_tag,
                    text=span.text
                ))
        
        return processed_spans
    
    def _spans_adjacent(self, span1: Span, span2: Span) -> bool:
        """Проверка, являются ли спаны соседними."""
        return span1.end == span2.start
    
    def process_spans(self, spans: List[Span]) -> List[ProcessedSpan]:
        """Полная обработка спанов: устранение конфликтов и валидация BIO."""
        # Устраняем конфликты
        resolved_spans = self.resolve_conflicts(spans)
        
        # Валидируем BIO-последовательность
        processed_spans = self.validate_bio_sequence(resolved_spans)
        
        # Сортируем по позиции
        processed_spans.sort(key=lambda x: x.start)
        
        return processed_spans
