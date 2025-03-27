# Human Action Recognition using CLIP

This project implements an end-to-end solution for Human Action Recognition (HAR) using CLIP (Contrastive Language-Image Pre-training) models. The system supports multiple training modes including single-GPU, DistributedDataParallel (DDP), and FullyShardedDataParallel (FSDP).

## Features

- **Advanced Model Architecture**: Leverages CLIP with custom text prompts for zero-shot and fine-tuned action classification
- **Distributed Training**: Supports single-GPU, DDP, and FSDP training modes
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and per-class accuracy visualizations
- **Dual Experiment Tracking**: Support for both MLflow (self-hosted) and Weights & Biases (cloud) simultaneously
- **Automated Training & Deployment**: End-to-end automated training with DVC dataset versioning and HuggingFace Hub integration
- **Production-Ready Inference**: REST API for model serving with multiple model format support (PyTorch, ONNX, TorchScript)
- **Data Version Control**: DVC integration for dataset and model versioning
- **Model Export**: ONNX and TensorRT export with benchmarking
- **Interactive UI**: Streamlit-based web interface for model testing

## System Architecture

The CLIP HAR project implements a comprehensive MLOps architecture with automation at its core, featuring interconnected components focused on human action recognition.

The system integrates:
- Data management with version control
- CLIP-based model architecture 
- Distributed training capabilities
- Comprehensive evaluation metrics
- Production-ready deployment options
- Interactive user interfaces

For detailed architecture diagrams and component descriptions, see:
- [System Architecture](docs/architecture.md)

## TensorRT Integration

To achieve maximum inference performance, I've integrated NVIDIA TensorRT into the project. TensorRT is a high-performance deep learning inference optimizer and runtime that significantly accelerates model inference on NVIDIA GPUs.

### Key TensorRT Features in This Project

- **GPU-Optimized Inference**: Up to 5x faster inference compared to standard PyTorch models
- **Multiple Precision Support**: FP32, FP16, and INT8 quantization options for optimal speed/accuracy tradeoffs
- **Dynamic Batch Processing**: Configurable batch sizes for both real-time and batch processing scenarios
- **Seamless Integration**: Works with the same API as other model formats (PyTorch, ONNX)

### How to Use TensorRT

#### 1. Export Your Model to TensorRT

```bash
python -m CLIP_HAR_PROJECT.deployment.export_clip_model \
    --model_path outputs/trained_model.pt \
    --config_path configs/training_config.yaml \
    --export_format tensorrt \
    --precision fp16 \  # Options: fp32, fp16, int8
    --batch_size 16 \
    --validate \
    --benchmark
```

#### 2. Serve the TensorRT Model

```bash
python -m CLIP_HAR_PROJECT.mlops.inference_serving \
    --model_path exports/model.trt \
    --model_type tensorrt \
    --class_names outputs/class_names.json \
    --port 8000
```

#### 3. Deploy with Docker

The provided Docker container has all necessary TensorRT dependencies pre-installed:

```bash
docker-compose -f docker/docker-compose.yml up clip-har-app
```

### TensorRT Performance

In my benchmarks, TensorRT provides significant speed improvements:

| Model Format | Inference Time (ms) | FPS | Relative Speed |
|--------------|---------------------|-----|----------------|
| PyTorch      | ~25-30ms            | ~35 | 1x             |
| ONNX         | ~15-20ms            | ~60 | ~1.7x          |
| TensorRT FP16| ~5-8ms              | ~150| ~4-5x          |

These improvements make real-time processing possible even on edge devices with compatible NVIDIA GPUs.

## Project Structure

```
CLIP_HAR_PROJECT/
├── app/                  # Streamlit application
├── configs/              # Configuration files
├── data/                 # Data handling modules
│   ├── dataset.py        # Dataset loading and preparation
│   ├── preprocessing.py  # Data preprocessing utilities
│   └── augmentation.py   # Augmentation strategies
├── deployment/           # Deployment utilities
│   └── export.py         # Model export (ONNX, TensorRT)
├── evaluation/           # Evaluation modules
│   ├── evaluator.py      # Evaluation orchestration
│   ├── metrics.py        # Metric computation utilities
│   └── visualization.py  # Result visualization
├── mlops/                # MLOps integration
│   ├── tracking.py       # Unified tracking system (MLflow & wandb)
│   ├── dvc_utils.py      # DVC integration utilities
│   ├── huggingface_hub_utils.py  # HuggingFace Hub integration
│   ├── automated_training.py     # Automated training module
│   └── inference_serving.py      # Inference serving module
├── models/               # Model definitions
│   ├── clip_model.py     # CLIP-based model
│   └── model_factory.py  # Model creation utilities
├── pipeline/             # End-to-end pipelines
│   ├── training_pipeline.py  # Training pipeline
│   └── inference_pipeline.py # Inference pipeline
├── training/             # Training modules
│   ├── distributed.py    # Distributed training utilities
│   └── trainer.py        # Trainer implementations
├── utils/                # Utility functions
├── docs/                 # Documentation
│   ├── architecture.md   # Detailed architecture overview
│   ├── docker_guide.md   # Docker containerization guide
│   ├── api_reference.md  # API reference documentation
│   └── experiment_tracking.md  # Experiment tracking guide
├── evaluate.py           # Evaluation script
├── launch_distributed.py # Distributed training launcher
├── train.py              # Training script
├── docker/               # Docker configuration files
│   ├── docker-compose.yml # Docker Compose configuration
│   ├── Dockerfile.train  # Training container Dockerfile
│   ├── Dockerfile.app    # App/Inference container Dockerfile
│   └── Dockerfile        # Base Dockerfile
├── dvc.yaml              # DVC pipeline definition
└── requirements.txt      # Project dependencies
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tuandung222/Open-vocabulary-Action-Recognition-with-CLIP.git
   cd CLIP_HAR_PROJECT
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up DVC:
   ```bash
   dvc init
   dvc add data/raw  # Add raw data to version control
   ```

## Training

### Single-GPU Training

```bash
python train.py --distributed_mode none --batch_size 128 --max_epochs 15
```

### Distributed Training (DDP)

```bash
python launch_distributed.py --distributed_mode ddp --batch_size 64 --max_epochs 15
```

### Using the Training Pipeline

```bash
python -m CLIP_HAR_PROJECT.pipeline.training_pipeline \
    --config_path configs/training_config.yaml \
    --output_dir outputs/training_run \
    --distributed_mode ddp \
    --augmentation_strength medium
```

### Automated Training

Automate the complete training pipeline with specific dataset versions and model checkpoints:

```bash
python -m CLIP_HAR_PROJECT.mlops.automated_training \
    --config configs/training_config.yaml \
    --output_dir outputs/auto_training \
    --dataset_version v1.2 \
    --checkpoint previous_models/checkpoint.pt \
    --push_to_hub \
    --experiment_name "clip_har_v2" \
    --distributed_mode ddp
```

The automated training pipeline:
- Loads a specific dataset version using DVC
- Starts from a checkpoint if provided
- Runs the complete training pipeline
- Pushes the trained model to HuggingFace Hub
- Saves all results and metrics

## Evaluation

Evaluate a trained model:

```bash
python evaluate.py --model_path /path/to/checkpoint.pt --output_dir results
```

The evaluation produces:
- Accuracy, precision, recall, and F1 score
- Confusion matrix visualization
- Per-class accuracy analysis
- Detailed classification report

## Experiment Tracking

### Dual Tracking with MLflow and Weights & Biases

The project supports both MLflow and Weights & Biases for experiment tracking simultaneously:

```bash
# Use both MLflow and wandb
python train.py --experiment_name "clip_har_experiment"

# Use only MLflow
python train.py --use_mlflow --no_wandb --experiment_name "clip_har_experiment"

# Use only wandb
python train.py --no_mlflow --use_wandb --experiment_name "clip_har_experiment"

# Disable all tracking
python train.py --no_tracking
```

### Setting up MLflow Server (Self-hosted)

```bash
# Start MLflow tracking server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

The MLflow UI (http://localhost:5000) provides:
- Experiment comparison
- Metric visualization
- Model versioning
- Artifact management

### Setting up Weights & Biases (Cloud)

```bash
# Login to wandb
wandb login

# Run training with wandb project/group
python train.py --use_wandb --project_name "clip_har" --group_name "experiments"
```

The wandb dashboard provides:
- Real-time training monitoring
- Advanced visualizations
- Team collaboration
- Run comparisons

## Data Version Control with DVC

```bash
# Initialize DVC
dvc init

# Add dataset to DVC tracking
dvc add data/raw

# Run the pipeline
dvc repro

# Push data to remote storage (if configured)
dvc push
```

The DVC pipeline in `dvc.yaml` includes stages for:
- Data preparation
- Model training
- Evaluation
- Model export

## Model Export and Deployment

Export a trained model:

```bash
python -m CLIP_HAR_PROJECT.deployment.export_clip_model \
    --model_path outputs/trained_model.pt \
    --export_format onnx tensorrt torchscript \
    --benchmark
```

## Docker Containers

The project provides Docker containers for training and inference:

```bash
# Build containers
docker-compose -f docker/docker-compose.yml build

# Run training container
docker-compose -f docker/docker-compose.yml run clip-har-train

# Run app/inference container
docker-compose -f docker/docker-compose.yml up clip-har-app
```

For detailed Docker setup, see [docs/docker_guide.md](docs/docker_guide.md).

## Inference Serving

Deploy a model as a REST API for inference:

```bash
# Serve a PyTorch model
python -m CLIP_HAR_PROJECT.mlops.inference_serving \
    --model_path outputs/trained_model.pt \
    --model_type pytorch \
    --port 8000

# Serve an ONNX model
python -m CLIP_HAR_PROJECT.mlops.inference_serving \
    --model_path outputs/model.onnx \
    --model_type onnx \
    --class_names outputs/class_names.json \
    --port 8001

# Serve a TorchScript model
python -m CLIP_HAR_PROJECT.mlops.inference_serving \
    --model_path outputs/model.torchscript \
    --model_type torchscript \
    --port 8002
```

### Using the Inference API

The inference API provides endpoints for:

1. **GET /** - Get service information
2. **GET /health** - Health check endpoint
3. **POST /predict** - Run inference on an image (JSON)
   - Image data as base64 string
   - Image URL
4. **POST /predict/image** - Run inference on an uploaded image (multipart/form-data)

Example API usage:

```python
# Using the Python client
from CLIP_HAR_PROJECT.mlops.inference_serving import InferenceClient

client = InferenceClient(url="http://localhost:8000")

# Predict from image file
result = client.predict_from_image_path("path/to/image.jpg")
print(f"Top prediction: {result['predictions'][0]['class_name']}")
print(f"Confidence: {result['predictions'][0]['score']:.4f}")

# Predict from image URL
result = client.predict_from_image_url("https://example.com/image.jpg")
```

For complete API reference, see [docs/api_reference.md](docs/api_reference.md).

## HuggingFace Hub Integration

Push trained models to HuggingFace Hub:

```python
from CLIP_HAR_PROJECT.mlops.huggingface_hub_utils import push_model_to_hub

# Push a trained model to HuggingFace Hub
model_url = push_model_to_hub(
    model=model,
    model_name="clip-har-v1",
    repo_id="tuandunghcmut/clip-har-v1",
    commit_message="Upload CLIP HAR model",
    metadata={"accuracy": 0.92, "f1_score": 0.91},
    private=False
)
print(f"Model uploaded to: {model_url}")
```

## Streamlit App

Run the Streamlit app:

```bash
streamlit run app/app.py
```

Features:
- Image upload for action classification
- Real-time webcam action recognition
- Model performance visualization

## Requirements

- Python 3.8+
- PyTorch 2.0+
- HuggingFace Transformers
- MLflow
- Weights & Biases
- DVC
- Streamlit
- ONNX Runtime
- FastAPI & Uvicorn (for inference serving)

## Dataset

The project uses the [Human Action Recognition (HAR) dataset](https://huggingface.co/datasets/Bingsu/Human_Action_Recognition) from HuggingFace, containing 15 action classes:

- calling
- clapping
- cycling
- dancing
- drinking
- eating
- fighting
- hugging
- laughing
- listening_to_music
- running
- sitting
- sleeping
- texting
- using_laptop

## Results

The CLIP-based model achieves:
- Zero-shot classification accuracy: ~81%
- Fine-tuned model accuracy: ~92%

## Documentation

- [Architecture Overview](docs/architecture.md)
- [Docker Setup Guide](docs/docker_guide.md)
- [API Reference](docs/api_reference.md)
- [Experiment Tracking Guide](docs/experiment_tracking.md)
- [Project Roadmap](ROADMAP.md)

## Kubernetes Deployment

For production deployments, I've prepared Kubernetes configurations to ensure scalable and reliable service operation. The deployment uses a microservices architecture with separate components for inference, model management, and monitoring.

### Kubernetes Setup

The `kubernetes/` directory contains all necessary configuration files:

```bash
# Apply the entire configuration
kubectl apply -f kubernetes/

# Or apply individual components
kubectl apply -f kubernetes/clip-har-inference.yaml
kubectl apply -f kubernetes/clip-har-monitoring.yaml
```

### Key Components

- **Inference Service**: Scalable pods with auto-scaling based on CPU/GPU utilization
- **Model Registry**: Persistent storage for model versions
- **API Gateway**: Manages external access and load balancing
- **HPA (Horizontal Pod Autoscaler)**: Automatically scales based on demand

### Resource Allocation

The deployment is configured with appropriate resource requests and limits:

```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1"
    nvidia.com/gpu: 1
  limits:
    memory: "4Gi"
    cpu: "2"
    nvidia.com/gpu: 1
```

## Monitoring Infrastructure

I've implemented a comprehensive monitoring stack using industry-standard tools for observability and performance tracking.

### Prometheus for Metrics Collection

The system uses Prometheus to collect and store time-series metrics:

- **Custom Metrics**: Model inference latency, throughput, GPU utilization
- **System Metrics**: Node resource utilization, network throughput
- **Business Metrics**: Requests per minute, success rates

```bash
# Access Prometheus dashboard
kubectl port-forward svc/prometheus-server 9090:9090
```

### Kibana and Elasticsearch for Logging

For log aggregation and analysis:

- **Centralized Logging**: All service logs collected and indexed
- **Structured Logging**: JSON-formatted logs with standardized fields
- **Log Retention**: Configurable retention policies

```bash
# Access Kibana dashboard
kubectl port-forward svc/kibana 5601:5601
```

### Grafana Dashboards

Preconfigured Grafana dashboards provide visual monitoring:

- **System Overview**: Resource utilization across the cluster
- **Model Performance**: Inference times, accuracy metrics
- **API Performance**: Request rates, latencies, error rates

```bash
# Access Grafana dashboard
kubectl port-forward svc/grafana 3000:3000
```

### Alerting

The monitoring system includes alerting for critical conditions:

- **Model Drift**: Alert when accuracy metrics drop below thresholds
- **Resource Constraints**: Notify on memory/CPU/GPU pressure
- **Error Rates**: Alert on elevated API error rates

Alerts can be configured to notify through various channels (email, Slack, PagerDuty).

## CI/CD Pipeline

The CLIP HAR project implements a robust CI/CD pipeline to automate testing, building, and deployment processes while ensuring code quality and operational reliability.

### Pipeline Overview

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    Build    │───▶│    Test     │───▶│ Model Eval  │───▶│  Artifacts  │───▶│   Deploy    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### GitHub Actions Workflows

All CI/CD processes are implemented using GitHub Actions, with separate workflows for different stages:

#### 1. Continuous Integration

```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install flake8 black isort
      - name: Run linters
        run: |
          flake8 .
          black --check .
          isort --check .
          
  test:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/
```

#### 2. Model Evaluation and Validation

```yaml
name: Model Evaluation

on:
  push:
    branches: [ main ]
    paths:
      - 'models/**'
      - 'training/**'
      - 'configs/**'

jobs:
  evaluate:
    runs-on: gpu-runner
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Download test dataset
        run: dvc pull data/test
      - name: Run model evaluation
        run: python evaluate.py --model_path outputs/trained_model.pt --test_data data/test
      - name: Upload metrics
        uses: actions/upload-artifact@v3
        with:
          name: model-metrics
          path: evaluation/metrics.json
```

#### 3. Container Build and Push

```yaml
name: Build and Push Containers

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      - name: Login to DockerHub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}
      - name: Build and push inference image
        uses: docker/build-push-action@v4
        with:
          context: .
          file: docker/Dockerfile.app
          push: true
          tags: tuandung222/clip-har-inference:latest
```

#### 4. Kubernetes Deployment

```yaml
name: Deploy to Kubernetes

on:
  workflow_run:
    workflows: ["Build and Push Containers"]
    types:
      - completed
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up kubectl
        uses: azure/setup-kubectl@v3
      - name: Configure kubeconfig
        run: |
          echo "${{ secrets.KUBE_CONFIG }}" > kubeconfig.yaml
          export KUBECONFIG=kubeconfig.yaml
      - name: Deploy to Kubernetes
        run: |
          kubectl apply -f kubernetes/clip-har-inference.yaml
          kubectl rollout status deployment/clip-har-inference
```

### Automated Model Retraining

The pipeline includes a scheduled job for model retraining:

```yaml
name: Scheduled Model Retraining

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday
  workflow_dispatch:

jobs:
  retrain:
    runs-on: gpu-runner
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Pull latest dataset
        run: dvc pull data
      - name: Run training
        run: python -m CLIP_HAR_PROJECT.mlops.automated_training --config configs/training_config.yaml
      - name: Push to model registry
        run: |
          python -m CLIP_HAR_PROJECT.mlops.huggingface_hub_utils --model_path outputs/model.pt
```

### GitOps with ArgoCD

For production environments, we use ArgoCD for GitOps-based continuous delivery:

1. Repository structure follows the GitOps pattern with environment-specific configurations
2. ArgoCD syncs the Kubernetes cluster state with the declared configurations
3. Promotion between environments (dev, staging, prod) via pull requests

### CI/CD Best Practices

- **Immutable Artifacts**: Container images are versioned and never modified
- **Canary Deployments**: New versions are deployed to a subset of users first
- **Automated Rollbacks**: Failed deployments trigger automatic rollbacks
- **Metric Validation**: Post-deployment checks verify system metrics
- **Security Scanning**: Container images are scanned for vulnerabilities

## Acknowledgements

- [OpenAI CLIP](https://github.com/openai/CLIP)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [Human Action Recognition Dataset](https://huggingface.co/datasets/Bingsu/Human_Action_Recognition)

## License

This project is licensed under the MIT License.
