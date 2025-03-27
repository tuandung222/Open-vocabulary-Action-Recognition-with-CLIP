FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app:${PYTHONPATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY requirements.txt .

# Install dependencies for inference and app
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn opencv-python streamlit

# Copy project code
COPY . .

# Create directory for model checkpoints
RUN mkdir -p /app/models/checkpoints

# Expose ports (8501 for Streamlit, 8000 for FastAPI)
EXPOSE 8501 8000

# Set health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health && \
        curl --fail http://localhost:8000/health || exit 1

# Default command - starts both the Streamlit app and FastAPI server
CMD ["sh", "-c", "nohup uvicorn CLIP_HAR_PROJECT.mlops.inference_serving:app --host 0.0.0.0 --port 8000 & streamlit run CLIP_HAR_PROJECT/app/app.py --server.port=8501 --server.address=0.0.0.0"]

# Labels for image metadata
LABEL maintainer="tuandunghcmut" \
      version="1.0" \
      description="CLIP-based Human Action Recognition - Demo App & Inference API"
