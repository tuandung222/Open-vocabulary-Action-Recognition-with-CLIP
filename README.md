# Human Action Recognition using CLIP

This project implements an end-to-end solution for Human Action Recognition (HAR) using CLIP (Contrastive Language-Image Pre-training) models. The system supports multiple training modes including single-GPU, DistributedDataParallel (DDP), and FullyShardedDataParallel (FSDP).

## Features

- **Advanced Model Architecture**: Leverages CLIP with custom text prompts for zero-shot and fine-tuned action classification
- **Distributed Training**: Supports single-GPU, DDP, and FSDP training modes
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and per-class accuracy visualizations
- **Dual Experiment Tracking**: Supports both MLflow (self-hosted) and Weights & Biases (cloud) simultaneously
- **Automated Training & Deployment**: End-to-end automated training with DVC dataset versioning and HuggingFace Hub integration
- **Production-Ready Inference**: REST API for model serving with multiple model format support (PyTorch, ONNX, TorchScript)
- **Data Version Control**: DVC integration for dataset and model versioning
- **Model Export**: ONNX and TensorRT export with benchmarking
- **Interactive UI**: Streamlit-based web interface for model testing

## System Architecture

The CLIP HAR project is structured in multiple layers:

1. **Data Layer**: Handles dataset loading, preprocessing, and augmentation
2. **Model Layer**: CLIP-based architecture with custom modifications
3. **Training Layer**: Distributed training with DDP and FSDP support
4. **Evaluation Layer**: Comprehensive metrics and visualizations
5. **Pipeline Layer**: End-to-end workflows including automated training
6. **MLOps Layer**: Experiment tracking and model versioning
7. **Deployment Layer**: Model export and inference serving
8. **Application Layer**: User interfaces including Streamlit and REST API

For a detailed architecture diagram, see [docs/architecture.md](docs/architecture.md).

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
├── docker-compose.yml    # Docker Compose configuration
├── Dockerfile.train      # Training container Dockerfile
├── Dockerfile.app        # App/Inference container Dockerfile
├── dvc.yaml              # DVC pipeline definition
└── requirements.txt      # Project dependencies
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/CLIP_HAR_PROJECT.git
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
docker-compose build

# Run training container
docker-compose run clip-har-train

# Run app/inference container
docker-compose up clip-har-app
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

## Acknowledgements

- [OpenAI CLIP](https://github.com/openai/CLIP)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [Human Action Recognition Dataset](https://huggingface.co/datasets/Bingsu/Human_Action_Recognition)

## License

This project is licensed under the MIT License. 