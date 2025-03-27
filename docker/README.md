# Docker Configuration

This directory contains Docker configuration files for the CLIP HAR project.

## Files

- `Dockerfile` - Base Dockerfile for development and simple deployments
- `Dockerfile.app` - Dockerfile for the inference and app container (includes TensorRT support)
- `Dockerfile.train` - Dockerfile for the training container (optimized for GPU training)
- `docker-compose.yml` - Docker Compose configuration for running the entire stack

## Usage

### Running with Docker Compose

From the project root directory:

```bash
# Build and start all services
docker-compose -f docker/docker-compose.yml up -d

# Build and start only the inference app
docker-compose -f docker/docker-compose.yml up -d clip-har-app

# Build and start only the training container
docker-compose -f docker/docker-compose.yml up -d clip-har-train
```

### Building Individual Containers

```bash
# Build the inference container
docker build -f docker/Dockerfile.app -t clip-har-app .

# Build the training container
docker build -f docker/Dockerfile.train -t clip-har-train .
```

## Container Details

### Training Container

The training container is built on PyTorch's CUDA-enabled image and includes:

- PyTorch with CUDA support
- DVC for data versioning
- MLflow and Weights & Biases for experiment tracking
- Additional training utilities

### Inference/App Container

The inference container includes:

- TensorRT support for optimized inference
- ONNX Runtime for running exported models
- FastAPI server for model serving API
- Streamlit for the interactive web UI

## Resource Requirements

Both containers are configured to use GPU resources when available. Modify the `deploy.resources` section in the docker-compose.yml file to adjust resource allocation. 