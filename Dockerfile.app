FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

LABEL maintainer="tuandung12092002@gmail.com"
LABEL version="1.0"
LABEL description="CLIP HAR Inference and App Container"

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:$PYTHONPATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3.8 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install TensorRT
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-libnvinfer=8.5.3-1+cuda11.7 \
    python3-libnvinfer-dev=8.5.3-1+cuda11.7 \
    libnvinfer8=8.5.3-1+cuda11.7 \
    libnvinfer-dev=8.5.3-1+cuda11.7 \
    libnvinfer-plugin8=8.5.3-1+cuda11.7 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for Python
RUN ln -sf /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Copy requirements.txt and install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install pycuda for TensorRT
RUN pip install --no-cache-dir pycuda==2022.2.2 nvidia-tensorrt==8.5.3.1

# Copy project code
COPY . /app/

# Create directory for model checkpoints
RUN mkdir -p /app/models/checkpoints

# Expose ports for Streamlit and FastAPI
EXPOSE 8501 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set default command to run both Streamlit app and FastAPI server
CMD ["sh", "-c", "uvicorn CLIP_HAR_PROJECT.mlops.inference_serving:app --host 0.0.0.0 --port 8000 & streamlit run CLIP_HAR_PROJECT/app/app.py"]
