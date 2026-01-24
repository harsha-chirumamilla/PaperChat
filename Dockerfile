# ==========================================
# Worker Base - System dependencies (CUDA)
# ==========================================
FROM nvidia/cuda:13.1.1-cudnn-runtime-ubuntu24.04 AS worker_base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-venv \
    python3-dev \
    curl \
    git \
    wget \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Ensure pip is installed for python3
RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*

# Symlink python to python3 for compatibility
RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Copy entrypoint script to both images
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# ==========================================
# API Stage - Lightweight (Python Slim)
# ==========================================
FROM python:3.12-slim AS api

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install basic utils if needed (curl for healthchecks)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install API dependencies (Latest)
COPY src/api/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy API and Common code
COPY src/api ./src/api
COPY src/common ./src/common

# Add source to python path
ENV PYTHONPATH=/app

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ==========================================
# Worker Stage - Heavy (ML Libraries)
# ==========================================
FROM worker_base AS worker



# Install Worker dependencies in a virtual environment with pip cache
COPY src/worker/requirements.txt requirements.txt

RUN python3 -m venv /app/venv \
    && /app/venv/bin/pip install --upgrade pip

RUN --mount=type=cache,target=/root/.cache/pip /app/venv/bin/pip install -r requirements.txt

# Ensure venv is used by default
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy All Code (Worker needs everything)
COPY src/worker ./src/worker
COPY src/rag ./src/rag
COPY src/common ./src/common

# Add source to python path
ENV PYTHONPATH=/app

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["celery", "-A", "src.worker.celery_app", "worker", "--loglevel=info"]
