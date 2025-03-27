# Human Action Recognition using CLIP

This project implements an end-to-end solution for Human Action Recognition (HAR) using CLIP (Contrastive Language-Image Pre-training) models. The system supports multiple training modes including single-GPU, DistributedDataParallel (DDP), and FullyShardedDataParallel (FSDP).

## 1. Features

- **Advanced Model Architecture**: Leverages CLIP with custom text prompts for zero-shot and fine-tuned action classification
- **Distributed Training**: Supports single-GPU, DDP, and FSDP training modes
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and per-class accuracy visualizations
- **Dual Experiment Tracking**: Support for both MLflow (self-hosted) and Weights & Biases (cloud) simultaneously
- **Automated Training & Deployment**: End-to-end automated training with DVC dataset versioning and HuggingFace Hub integration
- **Production-Ready Inference**: REST API for model serving with multiple model format support (PyTorch, ONNX, TorchScript)
- **Data Version Control**: DVC integration for dataset and model versioning
- **Model Export**: ONNX and TensorRT export with benchmarking
- **Interactive UI**: Streamlit-based web interface for model testing

## 2. System Architecture

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

## 3. Project Structure

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
├── custom_evaluate.py    # Evaluation script
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

## 4. Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tuandung222/Open-vocabulary-Action-Recognition-with-CLIP.git
   cd Open-vocabulary-Action-Recognition-with-CLIP
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. Set up DVC:
   ```bash
    # Intialize DVC if not already initialized
   dvc init
   dvc add data/raw  # Add raw data to version control
   ```

## 5. Training

### Single-GPU Training

For experiments and smaller datasets, run training on a single GPU:

```bash
python train.py --distributed_mode none --batch_size 128 --max_epochs 15 --lr 3e-6
```

### Distributed Training with DDP

For faster training with multiple GPUs using DistributedDataParallel (DDP):

```bash
# Launch DDP training using torchrun (automatically handles process creation)
python launch_distributed.py \
    --distributed_mode ddp \
    --batch_size 64 \
    --max_epochs 15 \
    --lr 3e-6 \
    --output_dir outputs/ddp_training
```

### Large-Scale Training with FSDP

For very large models or datasets, use Fully Sharded Data Parallel (FSDP) to shard model parameters across GPUs:

```bash
python launch_distributed.py \
    --distributed_mode fsdp \
    --batch_size 32 \
    --max_epochs 15 \
    --lr 2e-6 \
    --output_dir outputs/fsdp_training
```

### How Distributed Training Works

Under the hood, the distributed training in this project is implemented using PyTorch's distributed training capabilities, specifically DistributedDataParallel (DDP) and FullyShardedDataParallel (FSDP). Here's how it works:

1. **Launcher Abstraction**: When you run `launch_distributed.py`, it abstracts away the complexity of setting up distributed training:
   - It detects the number of available GPUs
   - It automatically configures the `torchrun` command with appropriate arguments
   - It launches the main training script (`train.py`) with the proper environment variables set

2. **Behind the Scenes**: The launcher is actually using `torchrun` (PyTorch's distributed launcher) to spawn multiple processes:
   ```python
   # From launch_distributed.py
   cmd = [
       "torchrun",
       "--nproc_per_node", str(num_gpus),
       "--master_addr", args.master_addr,
       "--master_port", args.master_port,
       "train.py",
       # ... additional arguments
   ]
   ```

3. **Process Management**: Each GPU gets its own Python process with:
   - A unique `LOCAL_RANK` (GPU index)
   - A unique `RANK` (process index in the distributed group)
   - Shared `WORLD_SIZE` (total number of processes)

4. **Trainer Integration**: Inside the `DistributedTrainer` class, the distributed environment is automatically set up based on these environment variables:
   - The model is wrapped in DDP or FSDP depending on your choice
   - Distributed samplers are created for the datasets
   - Gradients are synchronized across processes during training
   - Only the main process (rank 0) performs logging and checkpoint saving

This approach makes distributed training much simpler to use, as you don't have to manually set up process groups, wrap models, or handle synchronization - the launcher and trainer handle all these details for you.

### Training Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--distributed_mode` | Training mode (`none`, `ddp`, `fsdp`) | `none` |
| `--model_name` | CLIP model name/path | `openai/clip-vit-base-patch16` |
| `--batch_size` | Training batch size (per GPU) | 256 |
| `--eval_batch_size` | Evaluation batch size (per GPU) | 128 |
| `--max_epochs` | Maximum number of training epochs | 15 |
| `--lr` | Learning rate | 3e-6 |
| `--unfreeze_visual` | Unfreeze visual encoder | False |
| `--unfreeze_text` | Unfreeze text encoder | False |
| `--no_mixed_precision` | Disable mixed precision training | False |

### Using the Training Pipeline

Run an end-to-end training pipeline with all components:

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

For a comprehensive guide covering all training scenarios, distributed training options, and troubleshooting tips, see [Training Guide](docs/training_guide.md).

## 6. Evaluation

Evaluate a trained model:

```bash
python custom_evaluate.py --model_path /path/to/checkpoint.pt --output_dir results
```

The evaluation produces:
- Accuracy, precision, recall, and F1 score
- Confusion matrix visualization
- Per-class accuracy analysis
- Detailed classification report

## 7. Experiment Tracking

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

## 8. Data Version Control with DVC

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

## 9. Model Export and Deployment

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

## 10. TensorRT Integration

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

## 11. Docker Containers

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

## 12. Inference Serving

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

## 13. HuggingFace Hub Integration

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

### Advanced HuggingFace Hub Features

The project supports advanced HuggingFace Hub integration including:

- **Automated model publishing** during training
- **Custom model cards** with rich metadata
- **Complete pipeline publishing** for easier inference
- **CI/CD integration** through GitHub Actions
- **Model versioning** with tags and branches

For detailed instructions on these advanced features, see [HuggingFace Integration Guide](docs/huggingface_integration_guide.md).

## 14. Streamlit App

Run the Streamlit app:

```bash
streamlit run app/app.py
```

Features:
- Image upload for action classification
- Real-time webcam action recognition
- Model performance visualization

## 15. Requirements

- Python 3.8+
- PyTorch 2.0+
- HuggingFace Transformers
- MLflow
- Weights & Biases
- DVC
- Streamlit
- ONNX Runtime
- FastAPI & Uvicorn (for inference serving)

## 16. Dataset

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

## 17. Results

The CLIP-based model achieves:
- Zero-shot classification accuracy: ~81%
- Fine-tuned model accuracy: ~92%

## 18. Kubernetes Deployment

For production deployments, I've prepared Kubernetes configurations to ensure scalable and reliable service operation. The deployment uses a microservices architecture with separate components for inference, model management, and monitoring.

### Kubernetes Setup

The `kubernetes/`