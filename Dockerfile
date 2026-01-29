# ============================================
# Stage 1: Builder - Install dependencies
# ============================================
FROM python:3.12-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /build

# Copy requirements
COPY src/api/requirements.txt api-requirements.txt
COPY src/worker/requirements.txt worker-requirements.txt

# Create venv and install dependencies
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip wheel setuptools && \
    /opt/venv/bin/pip install --no-cache-dir -r api-requirements.txt && \
    /opt/venv/bin/pip install --no-cache-dir -r worker-requirements.txt

# ============================================
# Stage 2: Runtime - Minimal production image
# ============================================
FROM python:3.12-slim AS production

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    redis-server \
    supervisor \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy venv from builder
COPY --from=builder /opt/venv /app/venv

# Set environment variables
ENV VIRTUAL_ENV=/app/venv \
    PATH="/app/venv/bin:$PATH" \
    PYTHONPATH=/app

# Copy application code
COPY src ./src
COPY entrypoint.sh /app/entrypoint.sh
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Set permissions and create directories in one layer
RUN chmod +x /app/entrypoint.sh && \
    mkdir -p /var/log/supervisor /var/run /app/data/db /app/data/papers && \
    find /app/src -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /app/src -type f -name "*.pyc" -delete

# Expose port
EXPOSE 7860

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
