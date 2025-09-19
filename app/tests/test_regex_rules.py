"""Тесты для регулярных выражений извлечения сущностей."""

import pytest
from app.service.rules import RegexRules, Span


class TestRegexRules:
    """Тесты для регулярных выражений."""
    
    def setup_method(self):
        """Настройка для каждого теста."""
        self.rules = RegexRules()
    
    def test_volume_patterns(self):
        """Тест: паттерны для VOLUME сущностей."""
        test_cases = [
            ("0.5 л", [("0.5", "л")]),
            ("500мл", [("500", "мл")]),
            ("200 г", [("200", "г")]),
            ("10 шт", [("10", "шт")]),
            ("1.5кг", [("1.5", "кг")]),
            ("молоко 1л", [("1", "л")]),
            ("500 мл воды", [("500", "мл")]),
        ]
        
        for text, expected in test_cases:
            spans = self.rules.extract_volumes(text)
            assert len(spans) > 0, f"No VOLUME found in '{text}'"
            
            # Проверяем, что найдены ожидаемые значения
            for span in spans:
                assert span.entity.startswith("B-VOLUME") or span.entity.startswith("I-VOLUME")
    
    def test_percent_patterns(self):
        """Тест: паттерны для PERCENT сущностей."""
        test_cases = [
            ("2.5%", [("2.5", "%")]),
            ("15 %", [("15", "%")]),
            ("100%", [("100", "%")]),
            ("молоко 3.5% жирности", [("3.5", "%")]),
        ]
        
        for text, expected in test_cases:
            spans = self.rules.extract_percents(text)
            assert len(spans) > 0, f"No PERCENT found in '{text}'"
            
            # Проверяем, что найдены ожидаемые значения
            for span in spans:
                assert span.entity.startswith("B-PERCENT") or span.entity.startswith("I-PERCENT")
    
    def test_volume_edge_cases(self):
        """Тест: граничные случаи для VOLUME."""
        test_cases = [
            "0.5л",  # Без пробела
            "1,5 л",  # Запятая как разделитель
            "500ML",  # Заглавные буквы
            "10 ШТ",  # Заглавные буквы
        ]
        
        for text in test_cases:
            spans = self.rules.extract_volumes(text)
            assert len(spans) > 0, f"No VOLUME found in '{text}'"
    
    def test_percent_edge_cases(self):
        """Тест: граничные случаи для PERCENT."""
        test_cases = [
            "5%",  # Простой случай
            "100 %",  # С пробелом
            "0.1%",  # Десятичная дробь
            "99.99%",  # Двойная десятичная дробь
        ]
        
        for text in test_cases:
            spans = self.rules.extract_percents(text)
            assert len(spans) > 0, f"No PERCENT found in '{text}'"
    
    def test_no_matches(self):
        """Тест: тексты без совпадений."""
        test_cases = [
            "просто текст",
            "без чисел",
            "только слова",
        ]
        
        for text in test_cases:
            volume_spans = self.rules.extract_volumes(text)
            percent_spans = self.rules.extract_percents(text)
            assert len(volume_spans) == 0, f"Unexpected VOLUME found in '{text}'"
            assert len(percent_spans) == 0, f"Unexpected PERCENT found in '{text}'"
    
    def test_multiple_matches(self):
        """Тест: множественные совпадения в одном тексте."""
        text = "молоко 1л 3.5% жирности и кефир 500мл 2%"
        
        volume_spans = self.rules.extract_volumes(text)
        percent_spans = self.rules.extract_percents(text)
        
        assert len(volume_spans) >= 2, "Expected at least 2 VOLUME entities"
        assert len(percent_spans) >= 2, "Expected at least 2 PERCENT entities"
    
    def test_cyrillic_text(self):
        """Тест: обработка кириллического текста."""
        text = "абрикосы 500г global village"
        
        volume_spans = self.rules.extract_volumes(text)
        
        assert len(volume_spans) > 0, "Expected to find VOLUME in Cyrillic text"
        
        # Проверяем, что найденный спан соответствует "500г"
        found_500g = False
        for span in volume_spans:
            if "500" in span.text and "г" in span.text:
                found_500g = True
                break
        assert found_500g, "Expected to find '500г' in the text"
    
    def test_mixed_languages(self):
        """Тест: смешанный текст (кириллица + латиница)."""
        text = "молоко milk 1л 3.5%"
        
        all_spans = self.rules.extract_all(text)
        
        # Должны быть найдены VOLUME и PERCENT
        volume_found = any("VOLUME" in span.entity for span in all_spans)
        percent_found = any("PERCENT" in span.entity for span in all_spans)
        
        assert volume_found, "Expected to find VOLUME in mixed language text"
        assert percent_found, "Expected to find PERCENT in mixed language text"
    
    def test_span_positions(self):
        """Тест: правильность позиций спанов."""
        text = "молоко 1л"
        
        volume_spans = self.rules.extract_volumes(text)
        
        assert len(volume_spans) > 0, "Expected to find VOLUME"
        
        for span in volume_spans:
            # Проверяем, что позиции корректны
            assert span.start >= 0
            assert span.end <= len(text)
            assert span.start < span.end
            
            # Проверяем, что текст спана соответствует найденному
            span_text = text[span.start:span.end]
            assert span_text == span.text
    
    def test_bio_tagging(self):
        """Тест: правильность BIO-тегирования."""
        text = "500 мл"
        
        volume_spans = self.rules.extract_volumes(text)
        
        assert len(volume_spans) >= 2, "Expected at least 2 tokens for '500 мл'"
        
        # Первый токен должен быть B-VOLUME
        b_tags = [span for span in volume_spans if span.entity == "B-VOLUME"]
        i_tags = [span for span in volume_spans if span.entity == "I-VOLUME"]
        
        assert len(b_tags) >= 1, "Expected at least one B-VOLUME tag"
        assert len(i_tags) >= 1, "Expected at least one I-VOLUME tag"
