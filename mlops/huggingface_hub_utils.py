"""
HuggingFace Hub utilities for CLIP HAR project.

This module provides functions for pushing models, datasets, and metadata to
the HuggingFace Hub.
"""

import os
import json
import tempfile
import torch
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from huggingface_hub import HfApi, upload_folder, login, Repository
from transformers import CLIPModel, CLIPProcessor, AutoConfig

logger = logging.getLogger(__name__)


def login_to_hub(token: Optional[str] = None) -> None:
    """
    Login to HuggingFace Hub.

    Args:
        token: HuggingFace API token (if None, uses HF_TOKEN env variable)
    """
    if token is None:
        token = os.environ.get("HF_TOKEN")
        if token is None:
            raise ValueError(
                "No token provided. Set the HF_TOKEN environment variable or pass a token."
            )

    login(token=token)
    logger.info("Successfully logged in to HuggingFace Hub")


def push_model_to_hub(
    model: torch.nn.Module,
    model_name: str,
    repo_id: str,
    commit_message: str = "Upload model",
    model_card_content: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    private: bool = False,
    token: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Push a CLIP model to the HuggingFace Hub.

    Args:
        model: PyTorch model to push
        model_name: Name of the model (used for saving)
        repo_id: Repository ID (e.g. "username/model-name")
        commit_message: Commit message
        model_card_content: Content for the model card (README.md)
        metadata: Additional metadata for the model
        private: Whether the repository should be private
        token: HuggingFace API token (if None, uses HF_TOKEN env variable)
        config: Model configuration

    Returns:
        URL of the uploaded model
    """
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Save the model weights
        model_path = os.path.join(tmpdirname, "pytorch_model.bin")
        torch.save(model.state_dict(), model_path)

        # Create a config file
        if config is None:
            # If no config provided, create a default one
            if hasattr(model, "config"):
                model_config = model.config.to_dict()
            else:
                # Create a simple config
                model_config = {
                    "model_type": "clip",
                    "architectures": ["CLIPModel"],
                    "hidden_size": 768,
                    "intermediate_size": 3072,
                    "num_hidden_layers": 12,
                    "num_attention_heads": 12,
                }
        else:
            model_config = config

        # Add metadata if provided
        if metadata:
            model_config["metadata"] = metadata

        # Save config
        config_path = os.path.join(tmpdirname, "config.json")
        with open(config_path, "w") as f:
            json.dump(model_config, f, indent=2)

        # Create model card
        if model_card_content is None:
            model_card_content = f"""
# {model_name}

This is a CLIP-based model fine-tuned for Human Action Recognition (HAR).

## Usage

```python
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("{repo_id}")
processor = CLIPProcessor.from_pretrained("{repo_id}")

# Process image
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
```

## Training

This model was fine-tuned on the Human Action Recognition dataset.
"""

        model_card_path = os.path.join(tmpdirname, "README.md")
        with open(model_card_path, "w") as f:
            f.write(model_card_content)

        # Login to HuggingFace Hub
        login_to_hub(token)

        # Upload the folder
        repo_url = upload_folder(
            folder_path=tmpdirname,
            repo_id=repo_id,
            commit_message=commit_message,
            private=private,
        )

        logger.info(f"Model uploaded to {repo_url}")
        return repo_url


def push_pipeline_to_hub(
    pipeline: Any,
    repo_id: str,
    commit_message: str = "Upload pipeline",
    pipeline_card_content: Optional[str] = None,
    private: bool = False,
    token: Optional[str] = None,
) -> str:
    """
    Push a pipeline to the HuggingFace Hub.

    Args:
        pipeline: Pipeline to push
        repo_id: Repository ID (e.g. "username/pipeline-name")
        commit_message: Commit message
        pipeline_card_content: Content for the pipeline card (README.md)
        private: Whether the repository should be private
        token: HuggingFace API token (if None, uses HF_TOKEN env variable)

    Returns:
        URL of the uploaded pipeline
    """
    # Login to HuggingFace Hub
    login_to_hub(token)

    # Save and push pipeline
    if hasattr(pipeline, "save_pretrained"):
        # If pipeline has save_pretrained method, use it
        pipeline.save_pretrained(repo_id, push_to_hub=True, private=private)
        return f"https://huggingface.co/{repo_id}"
    else:
        # Otherwise, create a temporary directory and save manually
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Save the pipeline
            if hasattr(pipeline, "to_json"):
                pipeline_path = os.path.join(tmpdirname, "pipeline.json")
                with open(pipeline_path, "w") as f:
                    f.write(pipeline.to_json())
            elif hasattr(pipeline, "__dict__"):
                pipeline_path = os.path.join(tmpdirname, "pipeline.pth")
                torch.save(pipeline.__dict__, pipeline_path)
            else:
                pipeline_path = os.path.join(tmpdirname, "pipeline.pkl")
                import pickle

                with open(pipeline_path, "wb") as f:
                    pickle.dump(pipeline, f)

            # Create pipeline card
            if pipeline_card_content is None:
                pipeline_card_content = f"""
# CLIP HAR Pipeline

This is a pipeline for Human Action Recognition using CLIP.

## Usage

```python
from huggingface_hub import hf_hub_download
import pickle

# Download pipeline
pipeline_path = hf_hub_download(repo_id="{repo_id}", filename="pipeline.pkl")

# Load pipeline
with open(pipeline_path, "rb") as f:
    pipeline = pickle.load(f)

# Use pipeline
results = pipeline(image)
```
"""

            pipeline_card_path = os.path.join(tmpdirname, "README.md")
            with open(pipeline_card_path, "w") as f:
                f.write(pipeline_card_content)

            # Upload the folder
            repo_url = upload_folder(
                folder_path=tmpdirname,
                repo_id=repo_id,
                commit_message=commit_message,
                private=private,
            )

            logger.info(f"Pipeline uploaded to {repo_url}")
            return repo_url


def clone_repo(
    repo_id: str,
    local_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> str:
    """
    Clone a repository from HuggingFace Hub.

    Args:
        repo_id: Repository ID (e.g. "username/model-name")
        local_dir: Local directory to clone to
        token: HuggingFace API token (if None, uses HF_TOKEN env variable)

    Returns:
        Path to the cloned repository
    """
    # Login to HuggingFace Hub
    login_to_hub(token)

    # Create local directory if not provided
    if local_dir is None:
        local_dir = os.path.join(os.getcwd(), repo_id.split("/")[-1])

    # Clone repository
    repo = Repository(
        local_dir=local_dir,
        clone_from=repo_id,
        use_auth_token=token or os.environ.get("HF_TOKEN"),
    )

    logger.info(f"Repository cloned to {local_dir}")
    return local_dir


def create_model_card(
    model_name: str,
    repo_id: str,
    metrics: Optional[Dict[str, float]] = None,
    dataset_name: Optional[str] = None,
    model_description: Optional[str] = None,
) -> str:
    """
    Create a model card for HuggingFace Hub.

    Args:
        model_name: Name of the model
        repo_id: Repository ID (e.g. "username/model-name")
        metrics: Dictionary of metrics
        dataset_name: Name of the dataset
        model_description: Description of the model

    Returns:
        Model card content
    """
    if model_description is None:
        model_description = f"CLIP-based model fine-tuned for Human Action Recognition"

    model_card = f"""
# {model_name}

{model_description}

## Model Description

This model is fine-tuned on the {dataset_name or "Human Action Recognition"} dataset for classifying human actions in images.

## Intended Use & Limitations

This model is intended for classifying human actions in images. It may not perform well on images significantly different from the training data.

## Training Procedure

### Training Data

The model was trained on the {dataset_name or "Human Action Recognition"} dataset.

### Training Hyperparameters

* Learning rate: 5e-5
* Batch size: 32
* Optimizer: AdamW
* LR scheduler: Cosine with warmup
* Number of epochs: 10

## Evaluation Results

"""

    if metrics:
        model_card += "| Metric | Value |\n| ------ | ----- |\n"
        for metric, value in metrics.items():
            if isinstance(value, float):
                model_card += f"| {metric} | {value:.4f} |\n"
            else:
                model_card += f"| {metric} | {value} |\n"

    model_card += """
## Usage

```python
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("{{repo_id}}")
processor = CLIPProcessor.from_pretrained("{{repo_id}}")

# Process image
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
```
""".replace(
        "{{repo_id}}", repo_id
    )

    return model_card
