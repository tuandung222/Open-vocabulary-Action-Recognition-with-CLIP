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

## Model Export and Deployment

This project supports multiple export formats to optimize models for different deployment scenarios:

```bash
python -m CLIP_HAR_PROJECT.deployment.export_clip_model \
    --model_path outputs/trained_model.pt \
    --export_format onnx torchscript tensorrt \
    --benchmark
```

### Supported Export Formats

#### 1. ONNX
The Open Neural Network Exchange format provides cross-platform compatibility:
- Framework-independent model representation
- Optimized inference with ONNX Runtime
- Deployment on CPU, GPU, and specialized hardware

#### 2. TorchScript
PyTorch's serialization format for production deployment:
- C++ runtime compatibility
- Graph optimizations for faster inference
- Better portability than native PyTorch models

#### 3. TensorRT
NVIDIA's high-performance inference optimizer:
- Maximum GPU acceleration
- Mixed precision support (FP32, FP16, INT8)
- Kernel fusion and other advanced optimizations

### Performance Comparison

| Format      | Inference Time | FPS | Relative Speed | Use Case                |
|-------------|----------------|-----|----------------|-------------------------|
| PyTorch     | ~25-30ms       | ~35 | 1x             | Development, flexibility|
| TorchScript | ~18-22ms       | ~50 | ~1.4x          | Production CPU/GPU      |
| ONNX        | ~15-20ms       | ~60 | ~1.7x          | Cross-platform deploy   |
| TensorRT    | ~5-8ms         | ~150| ~4-5x          | Maximum GPU performance |

## TensorRT Integration

For applications requiring maximum inference speed, my TensorRT integration provides substantial performance benefits.

### Key TensorRT Features

- **GPU-Optimized Inference**: Up to 5x faster inference compared to standard PyTorch models
- **Multiple Precision Support**: FP32, FP16, and INT8 quantization options
- **Dynamic Batch Processing**: Configurable batch sizes for both real-time and batch processing
- **Seamless API Integration**: Uses the same inference API as other model formats

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

My CI/CD pipeline is implemented with GitHub Actions and consists of these key workflows:

#### 1. Continuous Integration
- **Trigger**: On push to main/develop branches and pull requests
- **Jobs**: Code linting (flake8, black, isort) and unit/integration tests
- **Benefits**: Ensures code quality and prevents breaking changes

#### 2. Model Evaluation
- **Trigger**: When model code changes are pushed
- **Jobs**: Pulls test data via DVC, evaluates model performance, uploads metrics
- **Hardware**: Runs on GPU-enabled self-hosted runners
- **Benefits**: Validates model performance before deployment

#### 3. Container Building
- **Trigger**: On pushes to main and version tags
- **Jobs**: Builds Docker images with optimized caching, pushes to DockerHub
- **Benefits**: Creates reproducible deployment artifacts

#### 4. Kubernetes Deployment
- **Trigger**: After successful container builds or manual dispatch
- **Jobs**: Applies Kubernetes configurations with rolling updates
- **Benefits**: Zero-downtime deployments with health checking

### Automated Model Retraining

The pipeline includes a weekly scheduled job for model retraining that:
- Pulls the latest dataset version from DVC
- Executes the automated training pipeline
- Pushes successful models to the model registry
- Can be manually triggered as needed

### GitOps with ArgoCD

For production environments, I use ArgoCD for GitOps-based continuous delivery:

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
