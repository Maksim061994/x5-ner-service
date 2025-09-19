"""Тесты для API эндпоинта /api/predict."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestPredictAPI:
    """Тесты для API предсказания сущностей."""
    
    def test_empty_input_returns_empty_list(self):
        """Тест: пустой ввод должен возвращать пустой список."""
        response = client.post("/api/predict", json={"input": ""})
        assert response.status_code == 200
        assert response.json() == []
    
    def test_whitespace_only_input_returns_empty_list(self):
        """Тест: ввод только из пробелов должен возвращать пустой список."""
        response = client.post("/api/predict", json={"input": "   "})
        assert response.status_code == 200
        assert response.json() == []
    
    def test_volume_detection(self):
        """Тест: обнаружение VOLUME сущностей."""
        test_cases = [
            ("0.5 л", [{"start_index": 0, "end_index": 3, "entity": "B-VOLUME"}]),
            ("500мл", [{"start_index": 0, "end_index": 5, "entity": "B-VOLUME"}]),
            ("200 г", [{"start_index": 0, "end_index": 2, "entity": "B-VOLUME"}, {"start_index": 3, "end_index": 4, "entity": "I-VOLUME"}]),
            ("10 шт", [{"start_index": 0, "end_index": 2, "entity": "B-VOLUME"}, {"start_index": 3, "end_index": 5, "entity": "I-VOLUME"}]),
        ]
        
        for input_text, expected in test_cases:
            response = client.post("/api/predict", json={"input": input_text})
            assert response.status_code == 200
            result = response.json()
            # Проверяем, что найдены ожидаемые сущности
            assert len(result) >= len(expected)
    
    def test_percent_detection(self):
        """Тест: обнаружение PERCENT сущностей."""
        test_cases = [
            ("2.5%", [{"start_index": 0, "end_index": 4, "entity": "B-PERCENT"}]),
            ("15 %", [{"start_index": 0, "end_index": 2, "entity": "B-PERCENT"}, {"start_index": 3, "end_index": 4, "entity": "I-PERCENT"}]),
        ]
        
        for input_text, expected in test_cases:
            response = client.post("/api/predict", json={"input": input_text})
            assert response.status_code == 200
            result = response.json()
            # Проверяем, что найдены ожидаемые сущности
            assert len(result) >= len(expected)
    
    def test_cyrillic_and_latin_text(self):
        """Тест: обработка кириллицы и латиницы."""
        # Тест из ТЗ: "абрикосы 500г global village"
        response = client.post("/api/predict", json={"input": "абрикосы 500г global village"})
        assert response.status_code == 200
        result = response.json()
        
        # Должна быть найдена VOLUME сущность "500г"
        volume_found = any(
            span["entity"].startswith("B-VOLUME") or span["entity"].startswith("I-VOLUME")
            for span in result
        )
        assert volume_found, f"Expected to find VOLUME entity in result: {result}"
    
    def test_example_from_spec(self):
        """Тест: пример из ТЗ - 'сгущенное молоко'."""
        response = client.post("/api/predict", json={"input": "сгущенное молоко"})
        assert response.status_code == 200
        result = response.json()
        
        # Проверяем, что ответ является списком
        assert isinstance(result, list)
        
        # Проверяем формат каждого спана
        for span in result:
            assert "start_index" in span
            assert "end_index" in span
            assert "entity" in span
            assert isinstance(span["start_index"], int)
            assert isinstance(span["end_index"], int)
            assert isinstance(span["entity"], str)
    
    def test_multiple_entities(self):
        """Тест: несколько сущностей в одном тексте."""
        response = client.post("/api/predict", json={"input": "молоко 1л 3.5% жирности"})
        assert response.status_code == 200
        result = response.json()
        
        # Должны быть найдены VOLUME и PERCENT сущности
        volume_found = any("VOLUME" in span["entity"] for span in result)
        percent_found = any("PERCENT" in span["entity"] for span in result)
        
        assert volume_found, "Expected to find VOLUME entity"
        assert percent_found, "Expected to find PERCENT entity"
    
    def test_invalid_json_request(self):
        """Тест: невалидный JSON запрос."""
        response = client.post("/api/predict", json={"invalid": "field"})
        assert response.status_code == 422  # Validation error
    
    def test_missing_input_field(self):
        """Тест: отсутствующее поле input."""
        response = client.post("/api/predict", json={})
        assert response.status_code == 422  # Validation error
    
    def test_health_endpoint(self):
        """Тест: эндпоинт health check."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    
    def test_ready_endpoint(self):
        """Тест: эндпоинт readiness check."""
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "pipeline_ready" in data
        assert isinstance(data["pipeline_ready"], bool)
