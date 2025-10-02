# X5 NER FastAPI Service

FastAPI-сервис для извлечения сущностей (NER) из поисковых запросов «Пятёрочки» с поддержкой BIO-разметки.

## Описание задачи

Сервис предназначен для извлечения сущностей из текстовых запросов пользователей с использованием BIO-разметки (B-ENTITY, I-ENTITY, O). 
Поддерживаются следующие типы сущностей:

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
- Устойчивость к параллельным нагрузкам

## Быстрый старт

### Локальный запуск

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd x5-ner-service
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

4. Запустите сервис:
```bash
# Для разработки
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Для продакшена с Gunicorn
gunicorn app.main:app -c gunicorn.conf.py
```

### Запуск с Docker Compose

```bash
# Сборка и запуск
docker-compose up -d

# Просмотр логов
docker-compose logs -f

# Остановка
docker-compose down
```

## Документация

### Основная документация
- [Модели и архитектура](models/README.md) - подробное описание ML-моделей
- [Датасеты](datasets/README.md) - описание данных для обучения


## Производственная конфигурация

### Преимущества настроенного Gunicorn

Сервис использует оптимизированную конфигурацию Gunicorn для максимальной производительности:

#### **Производительность**
- **Многопроцессность**: автоматический расчет количества воркеров на основе CPU
- **Многопоточность**: до 2 потоков на воркер для параллельной обработки
- **Preload**: предзагрузка приложения для ускорения старта воркеров
- **Keep-alive**: переиспользование соединений для снижения накладных расходов

#### **Надежность**
- **Graceful shutdown**: корректное завершение работы при получении сигналов
- **Health checks**: автоматическая проверка состояния сервиса
- **Restart policy**: автоматический перезапуск при сбоях
- **Resource limits**: ограничения на количество запросов для предотвращения утечек памяти

#### **Мониторинг**
- **Structured logging**: структурированные логи в JSON формате
- **Access logs**: детальная информация о запросах с временем выполнения
- **Metrics**: метрики производительности и использования ресурсов
- **Error tracking**: отслеживание ошибок и исключений

#### **Оптимизация для ML**
- **Thread management**: настройка количества потоков для ML библиотек
- **Memory optimization**: оптимизация использования памяти для больших моделей
- **CPU affinity**: привязка воркеров к CPU ядрам
- **Shared memory**: использование `/dev/shm` для временных файлов

### Конфигурация Gunicorn

```python
# Автоматический расчет ресурсов
workers = min(2, max(1, cpu_count // 2))
threads = max(2, cpu_count // workers)
worker_connections = 1000

# Оптимизация для ML
timeout = 900  # 15 минут для загрузки моделей
max_requests = 1000  # Перезапуск воркеров для предотвращения утечек
preload_app = True  # Предзагрузка приложения

# Настройки потоков для ML библиотек
os.environ["OMP_NUM_THREADS"] = str(threads)
os.environ["MKL_NUM_THREADS"] = str(threads)
```

## Архитектура

### Ансамблевая модель
Сервис использует комбинацию из трех моделей для максимальной точности:

1. **StackedCRF** - основная модель с ансамблем CRF классификаторов
2. **SpaCy Best** - быстрая модель с tok2vec эмбеддингами  
3. **SpaCy Full** - точная модель с transformer архитектурой

### Логика принятия решений
```python
if predict_crf != predict_spacy_best:
    predict_spacy_full = self._predict_spacy(text, name_model="full")
    if predict_spacy_full == predict_spacy_best:
        result = predict_spacy_best
    else:
        result = predict_crf  # по умолчанию
```

## Структура проекта

```
x5-ner-service/
├── app/                    # Основное приложение
│   ├── main.py            # Точка входа FastAPI
│   ├── api.py             # API роуты
│   ├── models.py          # Pydantic модели
│   ├── service/           # Бизнес-логика
│   │   ├── pipeline.py    # Основной пайплайн NER
│   │   ├── stacked_crf.py # StackedCRF модель
│   │   ├── features.py    # Извлечение признаков
│   │   ├── rules.py       # Регулярные выражения
│   │   └── postprocess.py # Постобработка спанов
│   └── core/              # Утилиты
│       ├── config.py      # Конфигурация
│       ├── logging.py     # Логирование
│       └── health.py       # Health checks
├── models/                 # ML модели
│   ├── crf/              # CRF модели
│   ├── spacy_best/       # SpaCy быстрая модель
│   └── spacy_full/       # SpaCy transformer модель
├── datasets/             # Данные для обучения
│   ├── data/             # CSV файлы
│   └── processing_*.py   # Скрипты обработки
├── research/             # Jupyter notebooks
├── scripts/             # Утилиты и тесты
├── gunicorn.conf.py     # Конфигурация Gunicorn
├── docker-compose.yml   # Docker Compose
└── Dockerfile           # Инструкции по сборке образа
```

## API Эндпоинты

### POST /api/predict
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

### GET /health
Проверка здоровья сервиса.

**Ответ:**
```json
{
  "status": "ok"
}
```

### GET /ready
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

### JavaScript

```javascript
// Fetch API
const response = await fetch('http://localhost:8000/api/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    input: 'сгущенное молоко'
  })
});

const result = await response.json();
console.log(result);
```

## Docker

### Сборка образа
```bash
docker build -t x5-ner-service .
```

### Запуск контейнера
```bash
docker run -p 8000:8000 \
  -v $(pwd)/models:/opt/models \
  -e MODELS_PATH=/opt/models \
  x5-ner-service
```

### Docker Compose
```bash
# Запуск всех сервисов
docker-compose up -d

# Просмотр логов
docker-compose logs -f x5_ner_service

# Масштабирование
docker-compose up -d --scale x5_ner_service=3
```

## Мониторинг и логирование

### Логи
- **Access logs**: информация о запросах и времени выполнения
- **Error logs**: ошибки и исключения
- **Application logs**: логи приложения с контекстом

### Метрики
- **Response time**: время ответа на запросы
- **Throughput**: количество запросов в секунду
- **Error rate**: процент ошибок
- **Memory usage**: использование памяти

### Health Checks
```bash
# Проверка здоровья
curl http://localhost:8000/health

# Проверка готовности
curl http://localhost:8000/ready
```

## Разработка

### Установка зависимостей
```bash
pip install -r requirements.txt
```

### Запуск тестов
```bash
pytest app/tests/
```

### Линтинг
```bash
# Black для форматирования
black app/

# isort для сортировки импортов
isort app/

# mypy для проверки типов
mypy app/
```

### Pre-commit hooks
```bash
pre-commit install
pre-commit run --all-files
```

## Производительность

### Бенчмарки
- **Среднее время ответа**: < 100ms
- **Пиковая нагрузка**: 1000 RPS
- **Использование памяти**: ~2GB на воркер
- **CPU utilization**: 60-80% при полной нагрузке

### Оптимизации
- **Кэширование моделей**: предзагрузка в память
- **Асинхронная обработка**: non-blocking I/O
- **Батчинг**: группировка запросов
- **Connection pooling**: переиспользование соединений

## Конфигурация

### Переменные окружения
```bash
# Путь к моделям
MODELS_PATH=/opt/models
```

### Конфигурационные файлы
- `gunicorn.conf.py` - настройки Gunicorn
- `logconf.ini` - конфигурация логирования Gunicorn
- `.env` - переменные окружения


## Вклад в проект

1. Fork репозитория
2. Создайте feature branch (`git checkout -b feature/amazing-feature`)
3. Commit изменения (`git commit -m 'Add amazing feature'`)
4. Push в branch (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

## Поддержка

Для вопросов и предложений создавайте Issues в репозитории.
