# Use official Python image
FROM python:3.8-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        git \
        && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir --progress-bar off --retries 10 --timeout 120 -r requirements.txt

# Copy the rest of the code
COPY . .

# Download models if not present (optional, assumes models.py handles this)
RUN python models.py || true

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI app with uvicorn
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000"]
