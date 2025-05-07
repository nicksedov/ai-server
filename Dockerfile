# Многоступенчатая сборка
FROM nvidia/cuda:12.8.1-base-ubuntu22.04

# Установка системных зависимостей
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip \
        curl \
        libgl1 \
        libglib2.0-0 && \        
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV STORAGE_ROOT=/app/storage \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:${PATH}" \
    HF_HOME=/app/storage/huggingface \
    HF_HUB_CACHE=/app/storage/huggingface

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    mkdir -p ${STORAGE_ROOT}/output

# Копирование исходного кода
COPY *.py .
COPY *.yaml .
COPY schemas/ ./schemas/
COPY routes/ ./routes/
COPY services/ ./services/
COPY resources/ ./resources/

EXPOSE 7999 11434

HEALTHCHECK --interval=60s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7999/v1/health -H "Content-Type: application/json" || exit 1

CMD ["python3", "main.py", "--config", "config-docker.yaml"]