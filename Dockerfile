# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy dependency files
COPY requirements.txt ./

# Install Python dependencies with uv
RUN uv pip install --system -r requirements.txt

RUN pip install huggingface_hub

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p cache/embeddings cache/chunks logs

# Expose port
EXPOSE 8000

# Run the application with hot reload for development
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
