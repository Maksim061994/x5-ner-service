# X5 NER FastAPI Service

FastAPI-сервис для извлечения сущностей (NER) из поисковых запросов «Пятёрочки» с поддержкой BIO-разметки.

## Описание задачи

Сервис предназначен для извлечения сущностей из текстовых запросов пользователей с использованием BIO-разметки (B-ENTITY, I-ENTITY, O). Поддерживаются следующие типы сущностей:

- **TYPE** - тип товара (например, "молоко", "хлеб")
- **BRAND** - бренд товара (например, "Домик в деревне", "Бородинский")
- **VOLUME** - объем/вес товара (например, "1л", "500г", "10шт")
- **PERCENT** - процентное содержание (например, "3.5%", "15%")

### Требования

- Публичный асинхронный эндпоинт `POST /api/predict`
- Формат запроса: `{"input": "<строка>"}`
- Формат ответа: список объектов `{start_index, end_index, entity}`
- При пустом `input` возвращается пустой список
- Лимит времени ответа: не более 1 секунды на запрос
- Сервис готов к деплою в Docker
- Устойчивость к параллельным нагрузкам

## Быстрый старт

### Локальный запуск

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd x5-ner-fastapi
```

2. Создайте виртуальное окружение:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

4. Скопируйте файл конфигурации:
```bash
cp env.example .env
```

5. Запустите сервис:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Запуск в Docker

1. Соберите Docker образ:
```bash
docker build -t x5-ner .
```

2. Запустите контейнер:
```bash
docker run -p 8000:8000 x5-ner
```

### Запуск с Docker Compose

```bash
docker-compose up --build
```

## API Документация

После запуска сервиса документация API доступна по адресам:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Основные эндпоинты

#### POST /api/predict
Извлечение сущностей из текста.

**Запрос:**
```json
{
  "input": "сгущенное молоко"
}
```

**Ответ:**
```json
[
  {
    "start_index": 0,
    "end_index": 8,
    "entity": "B-TYPE"
  },
  {
    "start_index": 9,
    "end_index": 15,
    "entity": "I-TYPE"
  }
]
```

#### GET /health
Проверка здоровья сервиса.

**Ответ:**
```json
{
  "status": "ok"
}
```

#### GET /ready
Проверка готовности сервиса к обработке запросов.

**Ответ:**
```json
{
  "status": "ok",
  "pipeline_ready": true
}
```

## Примеры использования

### cURL

```bash
# Базовый пример
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"input":"сгущенное молоко"}'

# Пример с объемом и процентом
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"input":"молоко 1л 3.5% жирности"}'

# Пример из ТЗ
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"input":"абрикосы 500г global village"}'
```

### Python

```python
import requests

# Базовый запрос
response = requests.post(
    "http://localhost:8000/api/predict",
    json={"input": "сгущенное молоко"}
)
print(response.json())

# Результат:
# [
#   {"start_index": 0, "end_index": 8, "entity": "B-TYPE"},
#   {"start_index": 9, "end_index": 15, "entity": "I-TYPE"}
# ]
```

## Интеграция с NER-моделью

Текущая реализация использует базовые регулярные выражения для извлечения VOLUME и PERCENT сущностей. Для TYPE и BRAND сущностей реализован плейсхолдер.

### Подключение реальной NER-модели

1. Установите дополнительные зависимости:
```bash
pip install transformers torch tokenizers
```

2. Раскомментируйте соответствующие строки в `requirements.txt`

3. Укажите имя модели в переменной окружения:
```bash
export MODEL_NAME="dbmdz/bert-base-russian-cased-finetuned-ner"
```

4. Модифицируйте файл `app/service/pipeline.py` для загрузки модели:

```python
from transformers import pipeline

async def _load_ner_model(self, model_name: str):
    """Загрузка NER-модели через HuggingFace Transformers."""
    self.ner_pipeline = pipeline(
        "ner", 
        model=model_name,
        aggregation_strategy="simple"
    )
```

### Рекомендуемые модели

Для русского языка рекомендуются следующие модели:
- `dbmdz/bert-base-russian-cased-finetuned-ner` - BERT для русского языка
- `DeepPavlov/rubert-base-cased` - RuBERT
- `sberbank-ai/ruBert-base` - Sberbank RuBERT

**Важно:** Не используйте коммерческие NER-API или закрытые датасеты. Используйте только открытые модели и данные.

## Тестирование

### Запуск тестов

```bash
# Все тесты
pytest

# С покрытием кода
pytest --cov=app --cov-report=html

# Конкретный тест
pytest app/tests/test_api_predict.py::TestPredictAPI::test_volume_detection
```

### Нагрузочное тестирование

```bash
# Простой нагрузочный тест
for i in {1..10}; do
  curl -X POST http://localhost:8000/api/predict \
    -H "Content-Type: application/json" \
    -d '{"input":"молоко 1л 3.5%"}' &
done
wait
```

## Конфигурация

Основные параметры конфигурации в файле `.env`:

```bash
# Сервер
PORT=8000
WORKERS=1
HOST=0.0.0.0

# Производительность
TIMEOUT_MS=900

# Модель
MODEL_NAME=

# Логирование
LOG_LEVEL=info
LOG_FORMAT=json

# Кэширование
CACHE_SIZE=1000
CACHE_TTL=3600
```

## Мониторинг и логирование

Сервис поддерживает:
- JSON-формат логов с корреляционными ID запросов
- Метрики производительности
- Health check эндпоинты
- Автоматическое логирование времени обработки запросов

## Разработка

### Установка зависимостей для разработки

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

### Линтинг и форматирование

```bash
# Проверка стиля кода
ruff check app/

# Форматирование кода
black app/

# Сортировка импортов
isort app/

# Проверка типов
mypy app/
```

### Pre-commit хуки

```bash
pip install pre-commit
pre-commit install
```

## Деплой

### Docker

```bash
# Сборка образа
docker build -t x5-ner:latest .

# Запуск в продакшене
docker run -d \
  --name x5-ner \
  -p 8000:8000 \
  -e LOG_LEVEL=info \
  -e TIMEOUT_MS=900 \
  x5-ner:latest
```

### Kubernetes

Пример манифеста для Kubernetes:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: x5-ner
spec:
  replicas: 3
  selector:
    matchLabels:
      app: x5-ner
  template:
    metadata:
      labels:
        app: x5-ner
    spec:
      containers:
      - name: x5-ner
        image: x5-ner:latest
        ports:
        - containerPort: 8000
        env:
        - name: LOG_LEVEL
          value: "info"
        - name: TIMEOUT_MS
          value: "900"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

## Лидерборд и Telegram-бот

Сервис должен быть доступен по публичному URL для оценки через Telegram-бота. Убедитесь, что:

1. Сервис доступен по публичному HTTPS URL
2. Эндпоинт `/api/predict` работает без авторизации
3. Сервис стабильно работает 24/7
4. Время ответа не превышает 1 секунду
5. Сервис корректно обрабатывает параллельные запросы

## Архитектура

```
app/
├── main.py              # Точка входа FastAPI
├── api.py               # API роуты
├── models.py            # Pydantic модели
├── service/             # Бизнес-логика
│   ├── pipeline.py      # Основной пайплайн NER
│   ├── rules.py         # Регулярные выражения
│   ├── tokenizer.py     # Токенизация
│   ├── postprocess.py   # Постобработка спанов
│   └── brands_types.py  # Детектор брендов/типов
├── core/                # Утилиты
│   ├── config.py        # Конфигурация
│   ├── logging.py       # Логирование
│   ├── timing.py        # Таймауты и метрики
│   └── health.py        # Health checks
└── tests/               # Тесты
    ├── test_api_predict.py
    ├── test_bio_merge.py
    └── test_regex_rules.py
```

## Лицензия

MIT License

## Поддержка

Для вопросов и предложений создавайте issues в репозитории проекта.
