# CLIP HAR Project Manual Testing Guide

This guide provides step-by-step instructions for manually testing all features of the CLIP HAR project in your local environment without Docker.

## Prerequisites

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/tuandung12092002/CLIP_HAR_PROJECT.git
   cd CLIP_HAR_PROJECT
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. Set up the Python path:
   ```bash
   # For Windows
   set PYTHONPATH=%cd%
   
   # For Linux/Mac
   export PYTHONPATH=$(pwd)
   ```

## 1. Data Processing Pipeline

### 1.1 Dataset Download and Exploration

```bash
# Create directory for dataset
mkdir -p data/raw

# Run the dataset download script
python -m CLIP_HAR_PROJECT.data.download \
    --dataset_name Bingsu/Human_Action_Recognition \
    --output_dir data/raw
```

### 1.2 Data Visualization and Exploration

```bash
# Run the dataset exploration script
python -m CLIP_HAR_PROJECT.data.explore \
    --dataset_dir data/raw \
    --save_dir data/exploration_results \
    --num_samples 5
```

Check that sample images for each class are saved in `data/exploration_results/`.

## 2. Training Models

### 2.1 Single GPU Training (Quick Test)

```bash
# Run a quick training with 2 epochs
python train.py \
    --distributed_mode none \
    --batch_size 64 \
    --max_epochs 2 \
    --output_dir outputs/test_run \
    --experiment_name test_run
```

Verify that:
- Training logs show progress
- Model checkpoints are saved in `outputs/test_run/checkpoints/`
- Training metrics are displayed

### 2.2 Training with Experiment Tracking

```bash
# Start MLflow server in a separate terminal
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# Run training with MLflow tracking
python train.py \
    --distributed_mode none \
    --batch_size 64 \
    --max_epochs 3 \
    --output_dir outputs/mlflow_test \
    --experiment_name mlflow_test \
    --use_mlflow

# Optional: Run with wandb (requires wandb account)
python train.py \
    --distributed_mode none \
    --batch_size 64 \
    --max_epochs 3 \
    --output_dir outputs/wandb_test \
    --experiment_name wandb_test \
    --use_wandb
```

Verify that:
- MLflow UI shows experiment at http://localhost:5000
- Training metrics and artifacts are logged

## 3. Model Evaluation

```bash
# Evaluate the trained model
python evaluate.py \
    --model_path outputs/test_run/checkpoints/best_model.pt \
    --output_dir evaluation_results/test_model \
    --batch_size 64
```

Verify that:
- Evaluation metrics (accuracy, precision, recall, F1) are displayed
- Confusion matrix is saved
- Per-class accuracy analysis is generated

## 4. Model Export

### 4.1 Export to ONNX

```bash
# Export model to ONNX format
python -m CLIP_HAR_PROJECT.deployment.export_clip_model \
    --model_path outputs/test_run/checkpoints/best_model.pt \
    --export_format onnx \
    --output_dir exports/onnx_model \
    --input_shape 3 224 224
```

### 4.2 Export to TorchScript

```bash
# Export model to TorchScript format
python -m CLIP_HAR_PROJECT.deployment.export_clip_model \
    --model_path outputs/test_run/checkpoints/best_model.pt \
    --export_format torchscript \
    --output_dir exports/torchscript_model \
    --input_shape 3 224 224
```

Verify that exported models are saved in the specified directories.

## 5. Inference Serving

### 5.1 Serve PyTorch Model

```bash
# Start inference server with PyTorch model
python -m CLIP_HAR_PROJECT.mlops.inference_serving \
    --model_path outputs/test_run/checkpoints/best_model.pt \
    --model_type pytorch \
    --class_names_path outputs/test_run/class_names.json \
    --port 8000
```

### 5.2 Serve ONNX Model

```bash
# Start inference server with ONNX model
python -m CLIP_HAR_PROJECT.mlops.inference_serving \
    --model_path exports/onnx_model/model.onnx \
    --model_type onnx \
    --class_names_path outputs/test_run/class_names.json \
    --port 8001
```

### 5.3 Test Inference API

In a separate terminal:
```bash
# Test the API with a sample image
curl -X POST http://localhost:8000/predict \
    -F "image=@data/raw/test_image.jpg"
```

Or using Python:
```python
from CLIP_HAR_PROJECT.mlops.inference_client import InferenceClient

# Test PyTorch model
client = InferenceClient(url="http://localhost:8000")
result = client.predict_from_image_path("data/raw/test_image.jpg")
print(f"PyTorch model prediction: {result}")

# Test ONNX model
client = InferenceClient(url="http://localhost:8001")
result = client.predict_from_image_path("data/raw/test_image.jpg")
print(f"ONNX model prediction: {result}")
```

## 6. Interactive App

```bash
# Start the Streamlit app
streamlit run CLIP_HAR_PROJECT/app/app.py
```

Verify in your browser (typically at http://localhost:8501):
- App loads correctly
- You can upload images for classification
- Real-time webcam functionality works (if available)
- Model performance visualization is displayed

## 7. End-to-End Workflow

### 7.1 Automated Training Pipeline

```bash
# Run the automated training pipeline
python -m CLIP_HAR_PROJECT.mlops.automated_training \
    --config configs/custom_config.py \
    --output_dir outputs/automated_run \
    --experiment_name automated_test \
    --max_epochs 3
```

### 7.2 HuggingFace Hub Integration (Optional)

```bash
# Login to HuggingFace Hub (requires HF account)
huggingface-cli login

# Push model to HuggingFace Hub
python -m CLIP_HAR_PROJECT.scripts.quick_train.py \
    --hub_repo_id your-username/clip-har-test \
    --experiment_name quick_test_hf
```

Verify that:
- Model is uploaded to HuggingFace Hub
- Model card is created with metrics

## 8. DVC Pipeline

```bash
# Initialize DVC
dvc init

# Add dataset to DVC
dvc add data/raw

# Create a DVC pipeline stage
dvc stage add -n prepare_data \
    -d data/raw \
    -o data/processed \
    python -m CLIP_HAR_PROJECT.data.preprocess

# Run the DVC pipeline
dvc repro
```

Verify that:
- DVC tracking is set up
- Pipeline stages run correctly

## 9. Verification Tests

Run the automated tests to verify everything is working:

```bash
# Run all tests
python -m CLIP_HAR_PROJECT.tests.run_tests --verbose

# Run specific test categories
python -m CLIP_HAR_PROJECT.tests.run_tests --type unit --verbose
python -m CLIP_HAR_PROJECT.tests.run_tests --type integration --verbose
```

## Troubleshooting

If you encounter issues:

1. Check the Python path:
   ```bash
   echo %PYTHONPATH%  # Windows
   echo $PYTHONPATH   # Linux/Mac
   ```

2. Verify dependencies are installed:
   ```bash
   pip list | grep -E "torch|transformers|mlflow|wandb|fastapi|streamlit"
   ```

3. Check log files:
   ```bash
   cat outputs/test_run/training.log
   ```

4. For GPU issues, verify CUDA availability:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"Device count: {torch.cuda.device_count()}")
   ``` 