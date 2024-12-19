FROM python:3.9-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Environment variables for memory optimization
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV MALLOC_TRIM_THRESHOLD_=100000
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV PYTHONUNBUFFERED=1
ENV PYTHONMALLOC=malloc
ENV MPLBACKEND=Agg

# Expose the port
EXPOSE 8000

# Command to run with memory optimizations
CMD ["gunicorn", "api.main:app", \
     "--workers", "1", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--worker-tmp-dir", "/dev/shm", \
     "--max-requests", "50", \
     "--max-requests-jitter", "10", \
     "--preload", \
     "--worker-connections", "50", \
     "--backlog", "50", \
     "--keep-alive", "2"]