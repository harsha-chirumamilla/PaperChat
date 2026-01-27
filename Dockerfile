FROM python:3.12-slim AS production

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies in one layer
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    poppler-utils \
    redis-server \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Install ALL Python dependencies (API + Worker)
COPY src/api/requirements.txt api-requirements.txt
COPY src/worker/requirements.txt worker-requirements.txt

# Create venv and install dependencies
RUN python3 -m venv /app/venv && \
    /app/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /app/venv/bin/pip install --no-cache-dir -r api-requirements.txt && \
    /app/venv/bin/pip install --no-cache-dir -r worker-requirements.txt

# Ensure venv is used by default
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy ALL application code
COPY src ./src

# Copy supervisord configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Create necessary directories
RUN mkdir -p /var/log/supervisor /var/run

# Add source to python path
ENV PYTHONPATH=/app

# Expose port for FastAPI
EXPOSE 7860

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
