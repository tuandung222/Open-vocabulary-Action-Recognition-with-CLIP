# Docker Setup Guide for CLIP HAR Project

This guide explains how to use Docker containers for the CLIP HAR project, including both the training container and the combined app/inference container.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU support)

## Project Container Structure

The project uses two main Docker containers:

1. **Training Container** (`clip-har-train`): For model training and evaluation
2. **App/Inference Container** (`clip-har-app`): Combined Streamlit UI and FastAPI inference service

Additionally, a local Docker registry is included for storing Docker images.

## Getting Started

### Building the Containers

```bash
# Build both containers
docker-compose build

# Build specific containers
docker-compose build clip-har-train
docker-compose build clip-har-app
```

### Running the Containers

#### Training Container

```bash
# Run with default parameters
docker-compose run clip-har-train

# Run with custom parameters
docker-compose run clip-har-train train.py --distributed_mode=ddp --batch_size=32 --max_epochs=20

# Run automated training with DVC and HuggingFace Hub
docker-compose run clip-har-train -m CLIP_HAR_PROJECT.mlops.automated_training \
    --config configs/training_config.yaml \
    --output_dir outputs/auto_training \
    --dataset_version v1.2 \
    --push_to_hub \
    --experiment_name "clip_har_v2"
```

#### App/Inference Container

```bash
# Run the combined app and inference service
docker-compose up clip-har-app

# Access the services:
# - Streamlit UI: http://localhost:8501
# - FastAPI Swagger UI: http://localhost:8000/docs
```

### Using the Local Registry

```bash
# Start the registry
docker-compose up -d registry

# Tag and push to local registry
docker tag clip-har-train:latest localhost:5000/clip-har-train:latest
docker push localhost:5000/clip-har-train:latest

docker tag clip-har-app:latest localhost:5000/clip-har-app:latest
docker push localhost:5000/clip-har-app:latest

# Pull from local registry
docker pull localhost:5000/clip-har-train:latest
docker pull localhost:5000/clip-har-app:latest
```

## Environment Variables

### Training Container

| Variable | Description | Default |
|----------|-------------|---------|
| `WANDB_API_KEY` | Weights & Biases API key | None |
| `HF_TOKEN` | HuggingFace API token | None |
| `CUDA_VISIBLE_DEVICES` | GPUs to use | 0 |

### App/Inference Container

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Path to the model checkpoint | `/app/models/checkpoints/best_model.pt` |
| `PYTHONPATH` | Python module path | `/app` |

## Volume Mounts

### Training Container

- `./data:/app/data` - Dataset directory
- `./configs:/app/configs` - Configuration files
- `./outputs:/app/outputs` - Output directory for results
- `${HOME}/.aws:/root/.aws:ro` - AWS credentials (for S3 access)
- `${HOME}/.dvc:/root/.dvc:ro` - DVC credentials

### App/Inference Container

- `./models/checkpoints:/app/models/checkpoints` - Model checkpoints
- `./data:/app/data:ro` - Dataset directory (read-only)
- `./evaluation_results:/app/evaluation_results` - Evaluation results

## GPU Support

The training container is configured to use GPUs with the NVIDIA Container Toolkit. To use GPUs:

1. Ensure the NVIDIA Container Toolkit is installed
2. Set the `CUDA_VISIBLE_DEVICES` environment variable to specify which GPUs to use

```bash
# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1 docker-compose run clip-har-train
```

## Customizing the Containers

### Extending the Training Container

To add additional packages to the training container:

```dockerfile
# Dockerfile.train.custom
FROM clip-har-train:latest

# Add custom packages
RUN pip install --no-cache-dir additional-package

# Override the default command
CMD ["train.py", "--custom-param", "value"]
```

### Extending the App/Inference Container

To add additional packages to the app container:

```dockerfile
# Dockerfile.app.custom
FROM clip-har-app:latest

# Add custom packages
RUN pip install --no-cache-dir additional-package

# Override the default command
CMD ["sh", "-c", "your-custom-command"]
```

## Deployment

For production deployment, consider:

1. Using a proper Docker registry (Docker Hub, AWS ECR, etc.)
2. Setting up proper authentication and secrets management
3. Using Kubernetes for orchestration
4. Implementing proper monitoring and logging

## Troubleshooting

### Common Issues

1. **GPU not available in container**: Ensure NVIDIA Container Toolkit is installed and configured
2. **Port conflicts**: Change the exposed ports in the `docker-compose.yml` file
3. **File permission issues**: Check volume mount permissions

### Logs

```bash
# View logs for app container
docker-compose logs clip-har-app

# View logs for training container
docker-compose logs clip-har-train
```

## Best Practices

1. **Tag your images** with version numbers for reproducibility
2. **Use environment variables** for configuration
3. **Mount volumes** for persistent data
4. **Use health checks** to ensure services are running properly
5. **Use Docker networks** to isolate services
