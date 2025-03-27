# CLIP HAR Project Architecture

This document provides a comprehensive overview of the CLIP HAR (Human Action Recognition) project architecture, detailing the various components and their interactions.

## System Overview

The CLIP HAR project is a comprehensive solution for human action recognition using the CLIP (Contrastive Language-Image Pre-training) model. The system architecture follows a modular design with clear separation of concerns, enabling flexibility and extensibility.

## Architectural Layers

The system is divided into seven main architectural layers:

### 1. Data Layer

The data layer manages all aspects of data handling, including:

- **Dataset Loading**: Loading and parsing of image datasets
- **Preprocessing**: Data normalization, resizing, and formatting
- **Augmentation**: Dynamic data augmentation for training robustness

**Key Components:**
- `data/dataset.py`: Dataset classes and loading utilities
- `data/preprocessing.py`: Data preprocessing pipelines
- `data/augmentation.py`: Augmentation strategies and transformations

### 2. Model Layer

The model layer contains the core model architectures and components:

- **CLIP Model**: Base CLIP architecture with custom modifications
- **Text Prompts**: Customizable text prompts for zero-shot and fine-tuning
- **Visual Encoder**: Image feature extraction
- **Text Encoder**: Text feature extraction
- **Multimodal Fusion**: Combining visual and text features

**Key Components:**
- `models/clip_model.py`: CLIP model implementation
- `models/model_factory.py`: Factory methods for model creation and configuration

### 3. Training Layer

The training layer handles all aspects of model training:

- **Trainer**: Core training loop and optimization
- **Distributed Training**: Support for DDP and FSDP
- **Optimization**: Learning rate scheduling, gradient clipping, etc.
- **Checkpointing**: Model saving and recovery

**Key Components:**
- `training/trainer.py`: Training loop implementation
- `training/distributed.py`: Distributed training utilities
- `launch_distributed.py`: Entry point for distributed training

### 4. Evaluation Layer

The evaluation layer manages model evaluation and performance analysis:

- **Evaluator**: Orchestrates evaluation workflows
- **Metrics Computation**: Calculating performance metrics
- **Visualization**: Visualizing results and model performance

**Key Components:**
- `evaluation/evaluator.py`: Evaluation orchestration
- `evaluation/metrics.py`: Metrics computation utilities
- `evaluation/visualization.py`: Visualization utilities
- `evaluate.py`: Entry point for standalone evaluation

### 5. Pipeline Layer

The pipeline layer connects the individual components into end-to-end workflows:

- **Training Pipeline**: Complete training workflow
- **Inference Pipeline**: End-to-end inference workflow
- **Batch Processing**: Efficient batch processing for large datasets
- **Automated Training**: Automated training with versioned datasets

**Key Components:**
- `pipeline/training_pipeline.py`: End-to-end training pipeline
- `pipeline/inference_pipeline.py`: Inference pipeline
- `mlops/automated_training.py`: Automated training module

### 6. MLOps Layer

The MLOps layer handles experiment tracking, model versioning, and deployment:

- **Experiment Tracking**: MLflow and Weights & Biases integration
- **DVC Versioning**: Dataset and model versioning
- **Model Registry**: Model storage and versioning
- **HuggingFace Hub**: Model sharing and distribution

**Key Components:**
- `mlops/tracking.py`: Unified tracking system for MLflow and wandb
- `mlops/dvc_utils.py`: DVC integration utilities
- `mlops/huggingface_hub_utils.py`: HuggingFace Hub integration

### 7. Deployment Layer

The deployment layer manages model export and deployment:

- **Model Export**: ONNX, TensorRT, and TorchScript export
- **Inference Serving**: FastAPI-based REST API
- **Containerization**: Docker containerization

**Key Components:**
- `deployment/export.py`: Model export utilities
- `mlops/inference_serving.py`: FastAPI-based inference service
- `Dockerfile.train` and `Dockerfile.app`: Docker containerization

### 8. Application Layer

The application layer provides user interfaces:

- **Streamlit App**: Interactive web UI for model testing
- **REST API**: API endpoints for model integration

**Key Components:**
- `app/app.py`: Streamlit application
- REST API endpoints in `mlops/inference_serving.py`

## Data Flow

The typical data flow through the system follows these steps:

1. Images are loaded and preprocessed in the data layer
2. The model layer processes images through the visual encoder
3. Text prompts are processed through the text encoder
4. Visual and text features are combined for classification
5. During training, results flow to the training layer
6. During evaluation, results flow to the evaluation layer
7. Results are tracked in the MLOps layer
8. Models are exported through the deployment layer
9. Users interact with the model through the application layer

## Component Interactions

### Training Workflow

1. `train.py` or `launch_distributed.py` initiates the training process
2. The training pipeline orchestrates the complete workflow
3. Data is loaded and preprocessed by the data layer
4. The model processes the data
5. The trainer optimizes the model parameters
6. Evaluation is performed periodically
7. Results are tracked via MLflow and/or wandb
8. Models are saved as checkpoints

### Inference Workflow

1. The user interacts with the app or API
2. The inference pipeline loads the model
3. Input data is preprocessed
4. The model makes predictions
5. Results are formatted and returned to the user

## Containerization Strategy

The project uses two main Docker containers:

1. **Training Container**: Focused on model training with GPU support
   - Uses `Dockerfile.train`
   - Contains all training dependencies
   - Configured for distributed training

2. **App/Inference Container**: Combined UI and API
   - Uses `Dockerfile.app`
   - Contains Streamlit and FastAPI
   - Serves both interactive UI and REST API endpoints

## Deployment Considerations

For production deployment, consider:

1. **Scaling**:
   - Training can scale with distributed training
   - Inference can scale with multiple containers

2. **Performance**:
   - Model export to ONNX or TensorRT for inference optimization
   - Batch processing for large-scale inference

3. **Monitoring**:
   - MLflow and W&B for experiment tracking
   - Container health checks for service monitoring

## Future Architecture Extensions

Planned architectural extensions include:

1. **Kubernetes Deployment**: For scalable deployment
2. **CI/CD Pipeline**: For automated testing and deployment
3. **Real-time Inference**: Optimized for video streams
4. **Transfer Learning Framework**: For custom datasets

## Technical Debt and Considerations

Current architectural considerations include:

1. Balancing flexibility vs. simplicity
2. Optimizing inference performance
3. Managing dependencies across containers
4. Ensuring reproducibility across environments

## Appendix: Directory Structure

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
├── evaluate.py           # Evaluation script
├── launch_distributed.py # Distributed training launcher
├── train.py              # Training script
├── docker-compose.yml    # Docker Compose configuration
├── Dockerfile.train      # Training container Dockerfile
├── Dockerfile.app        # App/Inference container Dockerfile
├── dvc.yaml              # DVC pipeline definition
└── requirements.txt      # Project dependencies
```
