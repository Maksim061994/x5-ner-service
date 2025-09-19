# Интеграция с NER-моделями

## Обзор

Реализована полная интеграция с NER-моделями через HuggingFace Transformers для извлечения брендов и типов товаров из текста.

## Установка зависимостей

```bash
pip install -r requirements.txt
```

Зависимости для NER-моделей уже включены в `requirements.txt`:
- `transformers==4.35.2`
- `torch==2.1.1`
- `tokenizers==0.15.0`

## Конфигурация

1. Скопируйте `env.example` в `.env`:
```bash
cp env.example .env
```

2. Укажите имя модели в файле `.env`:
```bash
MODEL_NAME=dbmdz/bert-large-cased-finetuned-conll03-english
```

## Рекомендуемые модели

### Для английского языка:
- `dbmdz/bert-large-cased-finetuned-conll03-english` - высокая точность
- `dslim/bert-base-NER` - быстрая, хорошая точность
- `Jean-Baptiste/camembert-ner` - для французского языка

### Для русского языка:
- `dbmdz/bert-base-russian-cased-finetuned-ner` - русская NER модель

## Использование

### Автоматическая инициализация

При запуске приложения NER-модель автоматически загружается, если указана в конфигурации:

```python
# В app/service/pipeline.py
if settings.MODEL_NAME and TRANSFORMERS_AVAILABLE:
    await self._load_ner_model(settings.MODEL_NAME)
    await self.brand_type_detector.initialize_ner_model(settings.MODEL_NAME)
```

### Извлечение сущностей

```python
# Извлечение брендов
brand_spans = await pipeline._extract_brands(text)

# Извлечение типов товаров
type_spans = await pipeline._extract_types(text)
```

## Архитектура

### Pipeline
- `_load_ner_model()` - загрузка NER-модели через HuggingFace Transformers
- `_extract_brands()` - извлечение брендов с фильтрацией по entity_group
- `_extract_types()` - извлечение типов товаров с фильтрацией по entity_group

### BrandTypeDetector
- `initialize_ner_model()` - инициализация NER-модели
- `detect_brands()` - обнаружение брендов в тексте
- `detect_types()` - обнаружение типов товаров в тексте

## Фильтрация сущностей

Модель фильтрует сущности по следующим критериям:

### Бренды:
- `entity_group` в `['BRAND', 'ORG', 'MISC']`
- Или содержит 'brand' в названии группы

### Типы товаров:
- `entity_group` в `['TYPE', 'PRODUCT', 'MISC']`
- Или содержит 'type' в названии группы

## Fallback режим

Если NER-модель недоступна или не указана:
- Используется только извлечение через регулярные выражения (VOLUME, PERCENT)
- Логируется предупреждение о недоступности NER-функциональности
- Приложение продолжает работать в базовом режиме

## Производительность

- Модели загружаются асинхронно в отдельном потоке
- Предсказания выполняются в thread pool executor
- Поддержка CPU и GPU (настраивается через параметр `device`)

## Мониторинг

Все операции логируются:
- Загрузка модели
- Ошибки при извлечении сущностей
- Предупреждения о недоступности библиотек

## Тестирование

Для тестирования без NER-модели просто не указывайте `MODEL_NAME` в конфигурации.
