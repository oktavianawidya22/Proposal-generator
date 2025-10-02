# ========= Base Image =========
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# OS deps (tesseract + langs) & tools
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      tesseract-ocr \
      tesseract-ocr-eng \
      tesseract-ocr-ind \
      libglib2.0-0 \
      libsm6 \
      libxrender1 \
      libxext6 \
      ca-certificates \
      && rm -rf /var/lib/apt/lists/*

# ========= App Layer =========
WORKDIR /app

# (opsional) pre-copy requirements for layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install -U pip && pip install -r requirements.txt

# Copy source
COPY app /app/app

# Expose port
EXPOSE 8000

# Env default (override di .env / Compose / host)
ENV OPENAI_MODEL=gpt-4o \
    OCR_LANG=eng

# Healthcheck sederhana
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1',8000)); s.close()"

# Start server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
