# Базовый образ с CUDA 12.8 и Ubuntu 22.04
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Настройка рабочей директории
WORKDIR /app

# Копирование файлов зависимостей
COPY requirements.txt .

# Установка Python пакетов
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY *.py .
COPY *.yaml .
COPY routes/*.py ./routes/
COPY services/*.py ./services/
COPY resources/*.joblib ./resources/

# Переменные окружения по умолчанию
ENV STORAGE_ROOT=/app/storage
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=${STORAGE_ROOT}/huggingface
ENV HUGGINGFACE_HUB_CACHE=${STORAGE_ROOT}/huggingface

# Создание директории для хранения данных
RUN mkdir -p ${STORAGE_ROOT}/output && \
    chmod -R 777 ${STORAGE_ROOT}

# Экспорт порта приложения
EXPOSE 7999
EXPOSE 11434

# Healthcheck для проверки доступности API
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7999/v1/health -H "Content-Type: application/json" || exit 1

# Команда запуска
CMD ["python3", "main.py", "--config", "config-docker.yaml"]