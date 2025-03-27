# Docker for ML Projects

## What is Docker?

Docker is a platform that allows you to package your application and all its dependencies into a standardized unit called a **container**. Think of a container as a lightweight, portable, self-sufficient package that includes everything needed to run your application:

- Code
- Runtime environment
- System libraries
- Dependencies
- Configuration files

## Why Docker is Essential for ML Projects

Machine learning projects often face the "it works on my machine" problem due to complex dependencies. Docker solves this by:

1. **Environment Consistency**: Ensuring the same environment across development, testing, and production
2. **Reproducibility**: Making it easy to recreate exact conditions for model training and inference
3. **Isolation**: Preventing conflicts between different projects and dependencies
4. **Portability**: Allowing models to run anywhere Docker is supported
5. **Scalability**: Making it easier to deploy models at scale

## Docker vs. Virtual Machines

Unlike virtual machines, Docker containers:
- Share the host OS kernel
- Start in seconds (not minutes)
- Use less memory and disk space
- Are more lightweight and efficient

## Key Docker Concepts

### 1. Images

A Docker **image** is like a template or blueprint for containers. It's a read-only file that contains:
- Base operating system
- Application code
- Libraries and dependencies
- Environment variables and settings

### 2. Containers

A Docker **container** is a running instance of an image. Containers:
- Are isolated from each other and the host system
- Can be started, stopped, moved, and deleted
- Can connect to networks and storage
- Maintain their own filesystems

### 3. Dockerfile

A **Dockerfile** is a text file with instructions for building an image. It defines:
- Base image to use
- Files to copy into the image
- Dependencies to install
- Commands to run
- Environment variables to set

### 4. Docker Compose

**Docker Compose** is a tool for defining and running multi-container Docker applications. Using a YAML file, you can:
- Configure multiple containers
- Define volumes and networks
- Specify environment variables
- Orchestrate startup/shutdown of services

## Docker Basics for ML

### Common Dockerfile Pattern for ML Projects

```dockerfile
# Start from a base image with Python and ML libraries
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set environment variables
ENV MODEL_PATH=/app/models/model.pt

# Command to run when container starts
CMD ["python", "serve_model.py"]
```

### Common docker-compose.yml Pattern

```yaml
version: '3'
services:
  training:
    build:
      context: .
      dockerfile: Dockerfile.train
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - CUDA_VISIBLE_DEVICES=0
  
  inference:
    build:
      context: .
      dockerfile: Dockerfile.inference
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
```

## Docker in Our CLIP HAR Project

In our project, we use Docker in two main ways:

### 1. Training Container

Our training container:
- Uses CUDA-enabled base image for GPU acceleration
- Installs PyTorch and other ML dependencies
- Mounts data and model directories for persistence
- Configures environment for reproducible training
- Integrates with experiment tracking tools

### 2. Inference/App Container

Our inference container:
- Provides a lightweight environment for model serving
- Exposes API endpoints for real-time inference
- Includes ONNX runtime for optimized performance
- Serves a Streamlit web interface
- Can be deployed to various environments

## Basic Docker Commands

```bash
# Build an image from a Dockerfile
docker build -t my-ml-model .

# Run a container from an image
docker run -p 8000:8000 my-ml-model

# List running containers
docker ps

# Stop a container
docker stop container_id

# View logs
docker logs container_id

# Run docker-compose
docker-compose up
```

## Benefits of Docker in Our Project

Docker helps us:

1. **Standardize Environments**: Ensuring consistency across development and deployment
2. **Manage GPU Dependencies**: Handling complex CUDA and cuDNN requirements
3. **Optimize for Different Stages**: Using a full environment for training but a lightweight one for deployment
4. **Support Multiple Export Formats**: Packaging different runtimes for ONNX, TensorRT, etc.
5. **Simplify Deployment**: Making it easy to run our models anywhere

## Docker Best Practices for ML

1. **Keep images slim**: Use multi-stage builds to reduce image size
2. **Manage data carefully**: Use volumes for datasets and model artifacts
3. **Layer caching**: Order Dockerfile commands to optimize build times
4. **Resource limits**: Set memory and CPU constraints
5. **Version pinning**: Specify exact versions of dependencies
6. **Health checks**: Add monitoring to ensure models are running properly

## Common Issues and Solutions

### GPU Access in Containers

To use GPUs in Docker:
```bash
docker run --gpus all my-ml-container
```

In docker-compose.yml:
```yaml
services:
  ml-service:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Permission Issues with Mounted Volumes

Run containers with the correct user ID:
```bash
docker run -u $(id -u):$(id -g) -v $(pwd):/app my-ml-container
```

### Networking Between Containers

Use Docker networks to allow containers to communicate:
```yaml
networks:
  ml-network:
    driver: bridge
```

## Further Reading

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Docker Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html)
- [Docker for Data Science](https://github.com/docker-for-data-science/docker-for-data-science-tutorial)

In the next guide, we'll explore CI/CD for ML projects and how to automate your ML workflows. 