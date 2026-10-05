# ── Multi-Platform Dockerfile for RAG Application ──
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure necessary runtime directories exist
RUN mkdir -p data/sample_documents vectorstore

# Expose Gradio port
EXPOSE 7860

# Launch application
CMD ["python", "app.py"]
