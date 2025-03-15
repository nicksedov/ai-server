# Базовый образ с CUDA 12.4 и Ubuntu 22.04
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

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
COPY . .

# Переменные окружения по умолчанию
ENV STORAGE_ROOT=/app/storage
ENV PYTHONUNBUFFERED=1

# Создание директории для хранения данных
RUN mkdir -p ${STORAGE_ROOT}/AI/output/flux && \
    chmod -R 777 ${STORAGE_ROOT}

# Экспорт порта приложения
EXPOSE 7999
EXPOSE 11434

# Healthcheck для проверки доступности API
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7999/v1/health -H "Content-Type: application/json" || exit 1

# Команда запуска
CMD ["python3", "main.py", "--config", "config-docker.yaml"]