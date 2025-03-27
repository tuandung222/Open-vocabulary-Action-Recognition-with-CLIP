# CLIP HAR Pipeline

This module contains end-to-end pipelines for training and inference in the CLIP HAR project.

## Overview

The pipeline module provides high-level abstractions for:

1. **Training Pipeline**: Orchestrates data preparation, model configuration, training, evaluation, and model export
2. **Inference Pipeline**: Handles input processing, model prediction, and output generation for single inputs
3. **Batch Inference Pipeline**: Processes large datasets in parallel, with support for images, videos, and CSV inputs

## Training Pipeline

The training pipeline handles the complete training workflow from data preparation to model export.

### Usage

```python
from CLIP_HAR_PROJECT.pipeline.training_pipeline import TrainingPipeline

# Create a pipeline with configuration
pipeline = TrainingPipeline(
    config_path="configs/training_config.yaml",
    output_dir="outputs/training_run",
    experiment_name="clip_har_experiment"
)

# Run the complete training process
pipeline.run()

# Or run individual steps
pipeline.prepare_data()
pipeline.setup_model()
pipeline.train()
pipeline.evaluate()
pipeline.export_model(export_format="onnx")
```

### Command Line Interface

```bash
# Run the complete training pipeline
python -m CLIP_HAR_PROJECT.pipeline.training_pipeline \
    --config configs/training_config.yaml \
    --output_dir outputs/training_run \
    --experiment_name clip_har_experiment

# Export a trained model to ONNX format
python -m CLIP_HAR_PROJECT.pipeline.training_pipeline \
    --config configs/training_config.yaml \
    --output_dir outputs/training_run \
    --export_only \
    --export_format onnx
```

## Inference Pipeline

The inference pipeline provides a simple interface for performing predictions with trained models.

### Usage

```python
from CLIP_HAR_PROJECT.pipeline.inference_pipeline import InferencePipeline

# Initialize the pipeline with a PyTorch model
pipeline = InferencePipeline(
    model_path="outputs/training_run/best_model.pt",
    model_type="pytorch",  # Options: pytorch, mlflow, onnx, tensorrt
    model_name="openai/clip-vit-base-patch16",
    device="cuda"
)

# Predict on a single image
result = pipeline.predict("path/to/image.jpg")
print(f"Predicted class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.4f}")

# Process a video
results = pipeline.predict_video(
    video_path="path/to/video.mp4",
    output_path="path/to/output.mp4",  # Optional: save annotated video
    frame_interval=5,                  # Process every 5 frames
    show_display=True                  # Show video during processing
)

# Clean up resources
pipeline.cleanup()
```

### Command Line Interface

```bash
# Run inference on a single image
python -m CLIP_HAR_PROJECT.pipeline.inference_pipeline \
    --model_path outputs/training_run/best_model.pt \
    --model_type pytorch \
    --image_path path/to/image.jpg

# Process a video with ONNX model
python -m CLIP_HAR_PROJECT.pipeline.inference_pipeline \
    --model_path outputs/training_run/model.onnx \
    --model_type onnx \
    --labels_file outputs/training_run/labels.json \
    --video_path path/to/video.mp4 \
    --output_path path/to/output.mp4 \
    --frame_interval 5
```

## Batch Inference Pipeline

The batch inference pipeline is designed for processing large datasets efficiently.

### Features

- Multi-threaded processing for high throughput
- Automatic GPU distribution when multiple GPUs are available
- Support for image directories, video directories, and CSV file inputs
- Result analysis and visualization

### Usage

```python
from CLIP_HAR_PROJECT.pipeline.batch_inference import BatchInferencePipeline

# Initialize the pipeline
pipeline = BatchInferencePipeline(
    model_path="outputs/training_run/model.onnx",
    model_type="onnx",
    labels_file="outputs/training_run/labels.json",
    batch_size=16,
    num_workers=4,
    device="cuda"
)

# Process a directory of images
results = pipeline.process_image_directory(
    input_dir="data/test_images",
    output_path="results/image_predictions.json",
    recursive=True
)

# Process a directory of videos
results = pipeline.process_video_directory(
    input_dir="data/test_videos",
    output_path="results/video_predictions.json",
    recursive=True,
    frame_interval=5,
    save_processed_videos=True
)

# Process images listed in a CSV file
results = pipeline.process_from_csv(
    csv_path="data/image_paths.csv",
    output_path="results/csv_predictions.json",
    path_column="image_path",
    base_dir="data"
)

# Analyze results
analysis = pipeline.analyze_results(
    results_path="results/image_predictions.json"
)
```

### Command Line Interface

```bash
# Process a directory of images
python -m CLIP_HAR_PROJECT.pipeline.batch_inference \
    --model_path outputs/training_run/model.onnx \
    --model_type onnx \
    --labels_file outputs/training_run/labels.json \
    --image_dir data/test_images \
    --output_path results/image_predictions.json \
    --recursive \
    --batch_size 32 \
    --num_workers 8 \
    --analyze_results

# Process videos
python -m CLIP_HAR_PROJECT.pipeline.batch_inference \
    --model_path outputs/training_run/model.onnx \
    --model_type onnx \
    --labels_file outputs/training_run/labels.json \
    --video_dir data/test_videos \
    --output_path results/video_predictions.json \
    --frame_interval 5 \
    --save_processed_videos

# Process from CSV
python -m CLIP_HAR_PROJECT.pipeline.batch_inference \
    --model_path outputs/training_run/model.onnx \
    --model_type onnx \
    --labels_file outputs/training_run/labels.json \
    --csv_path data/image_paths.csv \
    --csv_path_column image_path \
    --csv_base_dir data \
    --output_path results/csv_predictions.json
```

## Integration with DVC and MLflow

The pipelines integrate seamlessly with DVC and MLflow:

1. **DVC**: The training pipeline can be included in your DVC pipeline for reproducible machine learning workflows
2. **MLflow**: Both training and inference pipelines support MLflow tracking and model registry integration

### DVC Pipeline Example

```yaml
# dvc.yaml
stages:
  train:
    cmd: python -m CLIP_HAR_PROJECT.pipeline.training_pipeline --config configs/training_config.yaml --output_dir outputs/training_run
    deps:
      - data/processed
      - configs/training_config.yaml
    outs:
      - outputs/training_run/best_model.pt
      - outputs/training_run/metrics.json
    
  batch_inference:
    cmd: python -m CLIP_HAR_PROJECT.pipeline.batch_inference --model_path outputs/training_run/best_model.pt --image_dir data/test_images --output_path results/predictions.json
    deps:
      - outputs/training_run/best_model.pt
      - data/test_images
    outs:
      - results/predictions.json
``` 