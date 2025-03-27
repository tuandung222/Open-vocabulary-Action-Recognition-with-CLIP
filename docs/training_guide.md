# CLIP HAR Training Guide

This guide provides comprehensive instructions for training the CLIP HAR model using different distributed training approaches and automated workflows.

## Table of Contents

1. [Basic Setup](#basic-setup)
2. [Understanding Training Options](#understanding-training-options)
3. [Manual Training Workflows](#manual-training-workflows)
4. [Automated Training Pipeline](#automated-training-pipeline)
5. [Experiment Tracking](#experiment-tracking)
6. [Model Evaluation and Export](#model-evaluation-and-export)

## Basic Setup

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/tuandung222/Open-vocabulary-Action-Recognition-with-CLIP.git
cd CLIP_HAR_PROJECT

# Install dependencies
pip install -r requirements.txt

# Setup environment for distributed training
pip install torch torchvision torchaudio
```

### Project Structure

The training code is organized into several modules:
- `training/`: Core training utilities (trainer, distributed training)
- `pipeline/`: End-to-end training pipeline
- `mlops/`: MLOps utilities (tracking, automated training)
- `models/`: Model definitions and utilities
- `data/`: Data loading and preprocessing

## Understanding Training Options

### Single-GPU Training

Best for experimentation and small models. Uses a single GPU for training.

### Distributed Data Parallel (DDP)

- Replicates the model across multiple GPUs
- Each GPU processes a different batch of data
- Gradients are synchronized across GPUs
- Ideal for medium-sized models that fit on a single GPU

### Fully Sharded Data Parallel (FSDP)

- Shards model parameters, gradients, and optimizer states across GPUs
- Enables training larger models that wouldn't fit on a single GPU
- Reduces memory requirements at the cost of communication overhead
- Ideal for very large models

## Manual Training Workflows

### Single-GPU Training

```bash
# Basic command for single-GPU training
python train.py --distributed_mode none --batch_size 128 --max_epochs 15 --lr 3e-6
```

### Multi-GPU Training with DDP

```bash
# Launch distributed training with DDP
python launch_distributed.py \
    --distributed_mode ddp \
    --batch_size 64 \
    --max_epochs 15 \
    --lr 3e-6 \
    --output_dir outputs/ddp_training
```

### Large Model Training with FSDP

```bash
# Launch distributed training with FSDP
python launch_distributed.py \
    --distributed_mode fsdp \
    --batch_size 32 \
    --max_epochs 15 \
    --lr 2e-6 \
    --output_dir outputs/fsdp_training
```

### Important Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--distributed_mode` | Training mode (`none`, `ddp`, `fsdp`) | `none` |
| `--batch_size` | Training batch size (per GPU) | 256 |
| `--eval_batch_size` | Evaluation batch size (per GPU) | 128 |
| `--max_epochs` | Maximum number of training epochs | 15 |
| `--lr` | Learning rate | 3e-6 |
| `--model_name` | CLIP model name/path | `openai/clip-vit-base-patch16` |
| `--unfreeze_visual` | Unfreeze visual encoder | False |
| `--unfreeze_text` | Unfreeze text encoder | False |
| `--no_mixed_precision` | Disable mixed precision training | False |
| `--num_workers` | Number of dataloader workers | 2 |

### Fine-Tuning Strategies

#### 1. Zero-Shot Classification

Use CLIP without fine-tuning by just providing appropriate text prompts:

```bash
python train.py --evaluate_only --zero_shot --prompt_template "a photo of a person {label}"
```

#### 2. Linear Probing

Keep CLIP encoders frozen, only train a linear classification head:

```bash
python train.py --distributed_mode none --batch_size 128 --max_epochs 15 --lr 3e-6 
```

#### 3. Full Fine-Tuning

Unfreeze visual and/or text encoders for full fine-tuning:

```bash
python train.py --distributed_mode ddp --unfreeze_visual --batch_size 64 --max_epochs 20 --lr 1e-6
```

#### 4. Visual Encoder Only

Unfreeze only the visual encoder for more effective transfer learning:

```bash
python train.py --distributed_mode ddp --unfreeze_visual --batch_size 64 --max_epochs 15 --lr 2e-6
```

## Automated Training Pipeline

The CLIP HAR project includes a comprehensive automated training pipeline that handles dataset versioning, checkpointing, tracking, and more.

### Dataset Versioning with DVC

```bash
# Initialize DVC
dvc init

# Add dataset to DVC
dvc add data/raw

# Push dataset to remote storage
dvc push

# Pull specific dataset version
dvc pull --rev v1.2
```

### Automated Training Command

```bash
# Run automated training pipeline with DVC dataset version
python -m CLIP_HAR_PROJECT.mlops.automated_training \
    --config_path configs/training_config.yaml \
    --output_dir outputs/auto_training \
    --dataset_version v1.2 \
    --checkpoint_path previous_models/checkpoint.pt \
    --push_to_hub \
    --experiment_name "clip_har_v2" \
    --distributed_mode ddp
```

### Automated Training with Python API

```python
from CLIP_HAR_PROJECT.mlops.automated_training import run_automated_training

results = run_automated_training(
    config_path="configs/training_config.yaml",
    output_dir="outputs/auto_training",
    dataset_version="v1.2",
    checkpoint_path="previous_models/checkpoint.pt",
    push_to_hub=True,
    hub_repo_id="tuandunghcmut/clip-har-v2",
    experiment_name="clip_har_v2",
    distributed_mode="ddp"
)
```

## Experiment Tracking

The training pipeline supports dual tracking with MLflow and Weights & Biases.

### MLflow Tracking

```bash
# Start MLflow tracking server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# Run training with MLflow tracking
python train.py --use_mlflow --experiment_name "clip_har_experiment"
```

### Weights & Biases Integration

```bash
# Login to wandb
wandb login

# Run training with wandb tracking
python train.py --use_wandb --project_name "clip_har" --group_name "experiments"
```

### Tracked Metrics and Artifacts

The training pipeline tracks:
- Training/validation loss
- Accuracy, precision, recall, F1 score
- Confusion matrix
- Per-class accuracy
- Model parameters
- Exported models
- Training time

## Model Evaluation and Export

### Model Evaluation

```bash
# Evaluate a trained model
python evaluate.py --model_path outputs/trained_model.pt --output_dir results
```

### Exporting to Deployment Formats

The pipeline supports exporting to multiple formats:
- ONNX: Cross-platform inference
- TorchScript: Production deployment in PyTorch
- TensorRT: Maximum GPU performance

```bash
# Export model to multiple formats
python -m CLIP_HAR_PROJECT.deployment.export_clip_model \
    --model_path outputs/trained_model.pt \
    --export_format onnx torchscript tensorrt \
    --benchmark
```

### Advanced Configuration

For detailed customization, you can create YAML configuration files:

```yaml
# Example config.yaml
model:
  model_name: "openai/clip-vit-base-patch16"
  unfreeze_visual_encoder: true
  unfreeze_text_encoder: false

training:
  max_epochs: 20
  batch_size: 64
  lr: 2e-6
  mixed_precision: true
  distributed_mode: "ddp"

data:
  dataset_path: "data/raw"
  augmentation_strength: "medium"
  use_action_specific_augmentation: true
```

Use the configuration file in your training:

```bash
python -m CLIP_HAR_PROJECT.pipeline.training_pipeline --config_path configs/my_config.yaml
```

## Troubleshooting

### Common Issues with Distributed Training

1. **CUDA Out of Memory**
   - Reduce batch size
   - Enable mixed precision training
   - Switch to FSDP if model is too large
   
2. **Slow Training Speed**
   - Ensure GPUs are on the same node/machine for faster communication
   - Optimize DataLoader (increase num_workers, prefetch_factor)
   - Use mixed precision training

3. **Inconsistent Results**
   - Set a fixed random seed: `--seed 42`
   - Ensure identical initialization across processes

4. **Process Hanging**
   - Check for deadlocks in synchronization code
   - Ensure all processes call collective operations
   - Add appropriate timeout to distributed operations

### Best Practices

1. **Data Preparation**
   - Use appropriate augmentations for action recognition
   - Balance the dataset or use weighted sampling
   - Prefetch data to avoid I/O bottlenecks

2. **Model Selection**
   - Start with a pre-trained CLIP model
   - Experiment with different visual encoders (ViT-B/32, ViT-B/16, ViT-L/14)
   - Consider model size vs. available GPU memory

3. **Training Strategy**
   - Start with linear probing before full fine-tuning
   - Use learning rate warmup
   - Monitor validation metrics to avoid overfitting
   - Implement early stopping

4. **Production Deployment**
   - Export to appropriate format based on deployment target
   - Benchmark different formats to optimize inference speed
   - Consider model quantization for edge devices 