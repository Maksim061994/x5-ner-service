"""Тесты для BIO-сшивки и постобработки."""

import pytest
from app.service.postprocess import PostProcessor, ProcessedSpan
from app.service.rules import Span


class TestBIOMerge:
    """Тесты для BIO-сшивки и постобработки спанов."""
    
    def setup_method(self):
        """Настройка для каждого теста."""
        self.processor = PostProcessor()
    
    def test_bio_sequence_validation(self):
        """Тест: валидация BIO-последовательности."""
        # Создаем спаны для одной сущности
        spans = [
            Span(start=0, end=5, entity="B-VOLUME", text="500мл"),
            Span(start=6, end=8, entity="I-VOLUME", text="л"),
        ]
        
        processed = self.processor.validate_bio_sequence(spans)
        
        assert len(processed) == 2
        assert processed[0].entity == "B-VOLUME"
        assert processed[1].entity == "I-VOLUME"
    
    def test_conflict_resolution(self):
        """Тест: разрешение конфликтов между спанами."""
        # Создаем пересекающиеся спаны
        spans = [
            Span(start=0, end=5, entity="B-TYPE", text="молоко"),
            Span(start=2, end=7, entity="B-VOLUME", text="1л"),
        ]
        
        resolved = self.processor.resolve_conflicts(spans)
        
        # PERCENT > VOLUME > BRAND > TYPE, поэтому должен остаться только VOLUME
        assert len(resolved) == 1
        assert resolved[0].entity == "B-VOLUME"
    
    def test_priority_order(self):
        """Тест: правильный порядок приоритетов сущностей."""
        # PERCENT > VOLUME > BRAND > TYPE
        test_cases = [
            (Span(start=0, end=3, entity="B-PERCENT", text="5%"), Span(start=1, end=4, entity="B-VOLUME", text="1л")),
            (Span(start=0, end=3, entity="B-VOLUME", text="1л"), Span(start=1, end=4, entity="B-BRAND", text="brand")),
            (Span(start=0, end=3, entity="B-BRAND", text="brand"), Span(start=1, end=4, entity="B-TYPE", text="type")),
        ]
        
        for higher_priority, lower_priority in test_cases:
            resolved = self.processor.resolve_conflicts([higher_priority, lower_priority])
            assert len(resolved) == 1
            assert resolved[0].entity == higher_priority.entity
    
    def test_adjacent_spans(self):
        """Тест: обработка соседних спанов."""
        spans = [
            Span(start=0, end=3, entity="B-VOLUME", text="500"),
            Span(start=4, end=6, entity="I-VOLUME", text="мл"),
        ]
        
        processed = self.processor.validate_bio_sequence(spans)
        
        assert len(processed) == 2
        assert processed[0].entity == "B-VOLUME"
        assert processed[1].entity == "I-VOLUME"
    
    def test_non_adjacent_spans(self):
        """Тест: обработка несоседних спанов."""
        spans = [
            Span(start=0, end=3, entity="B-VOLUME", text="500"),
            Span(start=10, end=12, entity="B-VOLUME", text="мл"),  # Не соседний
        ]
        
        processed = self.processor.validate_bio_sequence(spans)
        
        assert len(processed) == 2
        assert processed[0].entity == "B-VOLUME"  # Первый спан
        assert processed[1].entity == "B-VOLUME"  # Второй спан тоже B-, так как не соседний
    
    def test_empty_spans(self):
        """Тест: обработка пустого списка спанов."""
        processed = self.processor.process_spans([])
        assert processed == []
    
    def test_single_span(self):
        """Тест: обработка одного спана."""
        spans = [Span(start=0, end=5, entity="B-VOLUME", text="500мл")]
        
        processed = self.processor.process_spans(spans)
        
        assert len(processed) == 1
        assert processed[0].entity == "B-VOLUME"
        assert processed[0].start == 0
        assert processed[0].end == 5
    
    def test_complex_scenario(self):
        """Тест: сложный сценарий с множественными конфликтами."""
        spans = [
            Span(start=0, end=5, entity="B-TYPE", text="молоко"),
            Span(start=6, end=8, entity="B-VOLUME", text="1л"),
            Span(start=7, end=10, entity="B-PERCENT", text="3.5%"),
            Span(start=11, end=16, entity="B-BRAND", text="brand"),
        ]
        
        processed = self.processor.process_spans(spans)
        
        # Должны остаться только PERCENT и BRAND (без конфликтов)
        assert len(processed) == 2
        entity_types = [span.entity.split('-')[1] for span in processed]
        assert "PERCENT" in entity_types
        assert "BRAND" in entity_types
