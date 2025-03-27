# HuggingFace Hub Integration Guide

This guide explains how to integrate the CLIP HAR project with HuggingFace Hub for model sharing and deployment.

## Setup HuggingFace Credentials

```bash
# Login to HuggingFace
huggingface-cli login

# Or use API token in automated scripts
export HUGGINGFACE_TOKEN="your_token_here"
```

## Basic Model Publishing

Push a trained model to HuggingFace Hub:

```python
from CLIP_HAR_PROJECT.mlops.huggingface_hub_utils import push_model_to_hub

# Push a trained model to HuggingFace Hub
model_url = push_model_to_hub(
    model=model,
    model_name="clip-har-v1",
    repo_id="your-username/clip-har-model",
    commit_message="Upload CLIP HAR model",
    metadata={"accuracy": 0.92, "f1_score": 0.91},
    private=False
)
print(f"Model uploaded to: {model_url}")
```

## Automated HuggingFace Upload During Training

```bash
# Include HuggingFace push in automated training
python -m CLIP_HAR_PROJECT.mlops.automated_training \
    --config_path configs/training_config.yaml \
    --output_dir outputs/training_run \
    --push_to_hub \
    --hub_repo_id "your-username/clip-har-model" \
    --private_repo False
```

## Customize Model Card and Metadata

Create a custom model card with detailed information:

```python
from CLIP_HAR_PROJECT.mlops.huggingface_hub_utils import push_model_to_hub, create_model_card

# Create custom model card
model_card = create_model_card(
    model_name="CLIP HAR Model",
    model_description="Human Action Recognition model using CLIP",
    usage_examples="""
    ```python
    from transformers import pipeline
    classifier = pipeline("image-classification", model="your-username/clip-har-model")
    result = classifier("path/to/image.jpg")
    ```
    """,
    model_architecture="CLIP-based classifier with fine-tuned vision encoder",
    training_data="Human Action Recognition dataset with 15 classes",
    training_metrics={"accuracy": 0.92, "f1_score": 0.91},
    limitations="May struggle with partially visible actions or unusual camera angles"
)

# Push model with custom card and rich metadata
model_url = push_model_to_hub(
    model=model,
    model_name="clip-har-v1",
    repo_id="your-username/clip-har-model",
    commit_message="Upload fine-tuned CLIP HAR model",
    metadata={
        "accuracy": 0.92, 
        "f1_score": 0.91,
        "dataset": "HAR-15",
        "training_type": "fine-tuned",
        "base_model": "openai/clip-vit-base-patch16",
        "training_epochs": 15,
        "framework": "PyTorch"
    },
    model_card=model_card,
    private=False
)
```

## Push Pipeline for Easy Inference

You can also push a complete inference pipeline:

```python
from CLIP_HAR_PROJECT.mlops.huggingface_hub_utils import push_pipeline_to_hub

# Push both model and inference pipeline
pipeline_url = push_pipeline_to_hub(
    model=model,
    tokenizer=tokenizer,
    image_processor=image_processor,
    repo_id="your-username/clip-har-pipeline",
    pipeline_type="image-classification",
    commit_message="Upload CLIP HAR inference pipeline",
    private=False
)
```

## Automate with CI/CD

Include HuggingFace Hub publishing in your GitHub Actions workflows:

```yaml
name: Train and Publish Model

on:
  workflow_dispatch:
  schedule:
    - cron: '0 0 * * 1'  # Weekly on Monday

jobs:
  train-and-publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Train and push model
        env:
          HF_TOKEN: ${{ secrets.HUGGINGFACE_TOKEN }}
        run: |
          python -m CLIP_HAR_PROJECT.mlops.automated_training \
            --config_path configs/training_config.yaml \
            --output_dir outputs/training_run \
            --push_to_hub \
            --hub_repo_id "your-username/clip-har-model" \
            --hub_token "$HF_TOKEN"
```

## Access Control and Collaboration

Manage access to your HuggingFace Hub models:

- **Private Models**: Set `private=True` when pushing to create private repositories
- **Team Access**: Use organization names in repo_id: `"your-org/model-name"`
- **Collaboration**: Add collaborators through the HuggingFace Hub UI
- **Versioning**: Use tags and branches for model versioning

## Advanced Features

### Model Versioning

```python
# Push model as a new version/tag
model_url = push_model_to_hub(
    model=model,
    model_name="clip-har-v1",
    repo_id="your-username/clip-har-model",
    commit_message="Release version 2.0",
    create_tag="v2.0",
    private=False
)
```

### API Endpoints

Once published, you can use your model through the HuggingFace Inference API:

```python
import requests

API_URL = f"https://api-inference.huggingface.co/models/your-username/clip-har-model"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

output = query("path/to/image.jpg")
```

### Spaces Demo

Create a HuggingFace Space to showcase your model with a web UI:

```python
from huggingface_hub import create_repo, upload_file

# Create a Streamlit demo space
space_repo_url = create_repo(
    repo_id="your-username/clip-har-demo",
    repo_type="space",
    space_sdk="streamlit",
    private=False
)

# Upload Streamlit app file
upload_file(
    path_or_fileobj="app/streamlit_demo.py",
    path_in_repo="app.py",
    repo_id="your-username/clip-har-demo",
    repo_type="space"
)
``` 