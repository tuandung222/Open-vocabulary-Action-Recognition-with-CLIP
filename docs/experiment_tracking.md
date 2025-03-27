# Experiment Tracking in CLIP HAR Project

This document explains how experiment tracking is implemented in the CLIP HAR project using both MLflow and Weights & Biases (wandb).

## Dual Tracking System

The CLIP HAR project implements a unified tracking system that supports both MLflow and Weights & Biases simultaneously. This dual approach provides flexibility and leverages the strengths of both platforms:

- **MLflow**: Self-hosted, comprehensive experiment tracking and model registry
- **Weights & Biases**: Cloud-based, rich visualizations and collaboration features

The dual tracking architecture is implemented through an abstraction layer in `mlops/tracking.py` that provides a consistent interface regardless of which tracking system(s) you use.

## Tracking Architecture

### Tracking Classes

The tracking system uses the following class hierarchy:

- **`ExperimentTracker`**: Abstract base class defining the tracking interface
- **`MLflowTracker`**: Implementation for MLflow
- **`WandbTracker`**: Implementation for Weights & Biases
- **`MultiTracker`**: Composite tracker that delegates to multiple trackers

### Key Features

- Start/end runs with automatic metadata collection
- Log parameters, metrics, and artifacts
- Track model performance and visualizations
- Unified interface regardless of backend
- Support for nested parameters and complex objects
- Integration with model registry
- Automatic model export for serving

## Setting Up Tracking

### MLflow Setup

1. **Start the MLflow server:**

   ```bash
   # Start MLflow tracking server
   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
   ```

2. **Access the MLflow UI:**

   Open your browser to `http://localhost:5000`

3. **Configure environment variables (optional):**

   ```bash
   export MLFLOW_TRACKING_URI=http://localhost:5000
   export MLFLOW_EXPERIMENT_NAME=clip_har_experiment
   ```

### Weights & Biases Setup

1. **Log in to wandb:**

   ```bash
   wandb login
   ```

2. **Set up your project (automatically done when using the wandb tracker):**

   ```bash
   # Optional manual setup
   wandb init
   ```

## Using the Tracking System

### Enabling Tracking from Command Line

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

### Programmatic Usage

```python
from CLIP_HAR_PROJECT.mlops.tracking import create_tracker

# Create a tracker with both MLflow and wandb
tracker = create_tracker(
    use_mlflow=True,
    use_wandb=True,
    experiment_name="clip_har_experiment",
    wandb_project="clip_har",
    wandb_group="experiments"
)

# Start a run
tracker.start_run(run_name="training_run_1")

# Log parameters
tracker.log_params({
    "learning_rate": 0.001,
    "batch_size": 32,
    "model_name": "clip-vit-base-patch16"
})

# Log metrics
for epoch in range(10):
    tracker.log_metrics({
        "train_loss": 0.1 - epoch * 0.01,
        "val_accuracy": 0.8 + epoch * 0.02
    }, step=epoch)

    # Log learning curve figure
    tracker.log_figure(fig, "learning_curve.png")

# Log confusion matrix
tracker.log_confusion_matrix(
    conf_matrix=conf_matrix,
    class_names=class_names,
    title="Confusion Matrix"
)

# Log model
tracker.log_model(
    model=model,
    artifact_path="models",
    signature=signature,
    input_example=example_input
)

# End the run
tracker.end_run()
```

## What Gets Tracked

The tracking system automatically logs:

### Parameters

- Model architecture details
- Training hyperparameters
- Data augmentation settings
- Optimizer and learning rate settings
- Hardware configuration

### Metrics

- Training/validation loss and accuracy
- Per-class precision, recall, F1 score
- Overall accuracy and weighted F1 score
- Inference time and throughput

### Artifacts

- Model checkpoints
- Confusion matrix visualizations
- Per-class accuracy charts
- Learning curves
- Example predictions
- Model exports (ONNX, TorchScript)

## Viewing Experiment Results

### MLflow UI

The MLflow UI provides:

1. **Experiment Comparison**: Compare parameters and metrics across runs
2. **Run Details**: View detailed information about individual runs
3. **Artifact Browsing**: Browse and download model artifacts
4. **Parameter Search**: Search for runs with specific parameters
5. **Metric Visualization**: Plot metrics across runs

### Weights & Biases Dashboard

The wandb dashboard provides:

1. **Real-time Monitoring**: View training progress in real-time
2. **Interactive Visualizations**: Explore performance metrics interactively
3. **Reports**: Create and share reports with team members
4. **System Monitoring**: Monitor resource usage during training
5. **Hyperparameter Visualization**: Visualize parameter importance

## Best Practices

1. **Use consistent experiment naming** to organize related runs
2. **Log all hyperparameters** at the start of training
3. **Log metrics at consistent intervals** (every epoch or validation step)
4. **Use tags** to categorize experiments
5. **Include sample inputs and outputs** for model interpretation
6. **Store model configurations** for reproducibility

## Local vs. Remote Tracking

The project supports both local and remote tracking:

### Local Tracking

- MLflow server running locally
- wandb in offline mode

### Remote Tracking

- MLflow server deployed on a remote server
- wandb connected to cloud account

To switch between local and remote:

```bash
# Remote MLflow
export MLFLOW_TRACKING_URI=http://remote-server:5000

# Local wandb (offline mode)
export WANDB_MODE=offline
```

## Integration with Model Registry

The tracking system integrates with model registries to version and manage models:

```python
# Register model in MLflow
model_info = tracker.register_model(
    model_name="clip_har_model",
    model_uri="runs:/last/models",
    description="CLIP HAR model trained on HAR dataset"
)

# Push model to HuggingFace Hub
tracker.push_to_hub(
    model=model,
    repo_id="organization/model-name",
    commit_message="Trained model with 92% accuracy"
)
```

## Experiment Tracking in Docker Containers

When running in Docker containers, experiment tracking is configured through environment variables:

```bash
# Set in docker/docker-compose.yml or when starting containers
docker run -e MLFLOW_TRACKING_URI=http://mlflow-server:5000 \
           -e WANDB_API_KEY=your-api-key \
           clip-har-train
```

The `docker/docker-compose.yml` file already includes appropriate volume mounts and environment variables for MLflow and wandb.
