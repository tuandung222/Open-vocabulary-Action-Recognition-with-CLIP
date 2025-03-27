"""
Automated training module for CLIP HAR Project.

This module provides functions for automated training workflows including:
- Loading specific dataset versions through DVC
- Loading checkpoint models
- Training models
- Pushing models to HuggingFace Hub
"""

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Union

import torch

from CLIP_HAR_PROJECT.mlops.huggingface_hub_utils import (
    create_model_card,
    push_model_to_hub,
    push_pipeline_to_hub,
)
from CLIP_HAR_PROJECT.pipeline.training_pipeline import TrainingPipeline

logger = logging.getLogger(__name__)


def setup_dvc(dvc_repo_url: Optional[str] = None) -> None:
    """
    Set up DVC for dataset versioning.

    Args:
        dvc_repo_url: URL of the DVC repository
    """
    # Check if DVC is installed
    try:
        subprocess.run(["dvc", "--version"], check=True, capture_output=True)
    except Exception as e:
        logger.error(f"DVC is not installed: {e}")
        logger.info("Install DVC with: pip install dvc")
        raise

    # Initialize DVC if needed
    if not os.path.exists(".dvc"):
        logger.info("Initializing DVC")
        subprocess.run(["dvc", "init"], check=True)

    # Add remote if provided
    if dvc_repo_url:
        logger.info(f"Adding DVC remote: {dvc_repo_url}")
        subprocess.run(
            ["dvc", "remote", "add", "-d", "storage", dvc_repo_url], check=True
        )


def get_dataset_version(
    version: str, data_path: str = "data", dvc_repo_url: Optional[str] = None
) -> str:
    """
    Get a specific version of the dataset using DVC.

    Args:
        version: Version tag or commit hash
        data_path: Path to the data directory
        dvc_repo_url: URL of the DVC repository

    Returns:
        Path to the dataset
    """
    # Setup DVC
    setup_dvc(dvc_repo_url)

    # Pull the specified version
    logger.info(f"Pulling dataset version: {version}")
    subprocess.run(["dvc", "checkout", version], check=True)
    subprocess.run(["dvc", "pull"], check=True)

    # Make sure the data path exists
    if not os.path.exists(data_path):
        raise ValueError(f"Data path {data_path} does not exist after DVC pull")

    return data_path


def load_checkpoint(
    checkpoint_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> torch.nn.Module:
    """
    Load a model checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on

    Returns:
        Loaded model
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint {checkpoint_path} does not exist")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            # Standard format with model state dict
            model_state = checkpoint["model"]

            # Try to get model class from checkpoint
            model_class = checkpoint.get("model_class", None)
            if model_class is not None:
                # If the checkpoint contains model class info, use it
                try:
                    if isinstance(model_class, str):
                        # Import the class dynamically
                        module_path, class_name = model_class.rsplit(".", 1)
                        module = __import__(module_path, fromlist=[class_name])
                        model_class = getattr(module, class_name)

                    # Create model instance
                    model = model_class()
                    model.load_state_dict(model_state)
                    return model
                except Exception as e:
                    logger.warning(
                        f"Failed to instantiate model using class from checkpoint: {e}"
                    )

        elif "state_dict" in checkpoint:
            # PyTorch Lightning format
            model_state = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            # Another common format
            model_state = checkpoint["model_state_dict"]
        else:
            # Assume the entire dict is the state dict
            model_state = checkpoint
    else:
        # Unexpected format
        raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")

    # We need to handle the case where we don't know the model class
    # This will be handled by the calling function which should know
    # which model to instantiate
    return model_state


def get_timestamp() -> str:
    """Get a timestamp string for naming purposes."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def run_automated_training(
    config_path: str,
    output_dir: str,
    dataset_version: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    push_to_hub: bool = False,
    hub_repo_id: Optional[str] = None,
    hub_token: Optional[str] = None,
    private_repo: bool = False,
    experiment_name: Optional[str] = None,
    distributed_mode: str = "none",
    use_mlflow: bool = True,
    use_wandb: bool = True,
) -> Dict[str, Any]:
    """
    Run an automated training pipeline.

    Args:
        config_path: Path to the configuration file
        output_dir: Directory to save outputs
        dataset_version: Version of the dataset to use
        checkpoint_path: Path to a checkpoint to continue training from
        push_to_hub: Whether to push the model to HuggingFace Hub
        hub_repo_id: Repository ID for HuggingFace Hub
        hub_token: HuggingFace API token
        private_repo: Whether to create a private repository
        experiment_name: Name for the experiment
        distributed_mode: Training mode (none, ddp, fsdp)
        use_mlflow: Whether to use MLflow for tracking
        use_wandb: Whether to use wandb for tracking

    Returns:
        Dictionary with training results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set experiment name if not provided
    if experiment_name is None:
        timestamp = get_timestamp()
        experiment_name = f"automated_training_{timestamp}"

    # Get dataset version if specified
    if dataset_version:
        data_path = get_dataset_version(dataset_version)
        logger.info(f"Using dataset version {dataset_version} at {data_path}")

    # Create training pipeline
    logger.info(f"Creating training pipeline with config {config_path}")
    pipeline = TrainingPipeline(
        config_path=config_path,
        distributed_mode=distributed_mode,
        use_mlflow=use_mlflow,
        use_wandb=use_wandb,
        experiment_name=experiment_name,
    )

    # Load checkpoint if specified
    if checkpoint_path:
        logger.info(f"Loading checkpoint {checkpoint_path}")
        checkpoint_state = load_checkpoint(checkpoint_path)

        # Ensure we have a model setup
        if pipeline.model is None:
            pipeline.setup_model()

        # Load state dict
        if isinstance(checkpoint_state, dict):
            pipeline.model.load_state_dict(checkpoint_state)
        else:
            # Assume checkpoint_state is already a model
            pipeline.model = checkpoint_state

    # Run the training pipeline
    logger.info("Starting training run")
    results = pipeline.run()

    # Push to HuggingFace Hub if requested
    if push_to_hub and results:
        if hub_repo_id is None:
            if experiment_name:
                hub_repo_id = f"tuandunghcmut/clip-har-{experiment_name.replace(' ', '-').lower()}"
            else:
                timestamp = get_timestamp()
                hub_repo_id = f"tuandunghcmut/clip-har-{timestamp}"

        logger.info(f"Pushing model to HuggingFace Hub: {hub_repo_id}")

        # Create model card with metrics
        metrics = results.get("evaluation_metrics", {})
        model_card = create_model_card(
            model_name=hub_repo_id.split("/")[-1],
            repo_id=hub_repo_id,
            metrics=metrics,
            dataset_name="Human Action Recognition",
        )

        # Get training config for metadata
        training_config = {}
        if hasattr(pipeline, "config"):
            if hasattr(pipeline.config, "training"):
                training_config = pipeline.config.training.__dict__

        # Push model to hub
        hub_url = push_model_to_hub(
            model=pipeline.model,
            model_name=hub_repo_id.split("/")[-1],
            repo_id=hub_repo_id,
            commit_message=f"Upload model from automated training: {experiment_name}",
            model_card_content=model_card,
            metadata={
                "training_config": training_config,
                "metrics": {
                    k: float(v) if isinstance(v, (int, float)) else str(v)
                    for k, v in metrics.items()
                    if not isinstance(v, (list, dict, set))
                },
                "training_timestamp": get_timestamp(),
            },
            private=private_repo,
            token=hub_token,
        )

        # Add hub URL to results
        results["hub_url"] = hub_url

    return results


def parse_args():
    """Parse command line arguments for automated training."""
    parser = argparse.ArgumentParser(description="Run automated training for CLIP HAR")

    # Required arguments
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save outputs"
    )

    # Optional arguments
    parser.add_argument(
        "--dataset_version",
        type=str,
        default=None,
        help="Version of the dataset to use",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to continue training from",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the trained model to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub_repo_id",
        type=str,
        default=None,
        help="Repository ID for HuggingFace Hub",
    )
    parser.add_argument(
        "--private_repo",
        action="store_true",
        help="Create a private repository on HuggingFace Hub",
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None, help="Name for the experiment"
    )
    parser.add_argument(
        "--distributed_mode",
        type=str,
        default="none",
        choices=["none", "ddp", "fsdp"],
        help="Training mode (none, ddp, fsdp)",
    )
    parser.add_argument(
        "--no_mlflow", action="store_true", help="Disable MLflow tracking"
    )
    parser.add_argument(
        "--no_wandb", action="store_true", help="Disable wandb tracking"
    )

    return parser.parse_args()


def main():
    """Main function to run automated training from command line."""
    # Parse arguments
    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.output_dir, "training.log")),
        ],
    )

    # Run automated training
    try:
        results = run_automated_training(
            config_path=args.config,
            output_dir=args.output_dir,
            dataset_version=args.dataset_version,
            checkpoint_path=args.checkpoint,
            push_to_hub=args.push_to_hub,
            hub_repo_id=args.hub_repo_id,
            private_repo=args.private_repo,
            experiment_name=args.experiment_name,
            distributed_mode=args.distributed_mode,
            use_mlflow=not args.no_mlflow,
            use_wandb=not args.no_wandb,
        )

        # Save results
        results_path = os.path.join(args.output_dir, "results.json")
        with open(results_path, "w") as f:
            # Convert non-serializable values to strings
            serializable_results = {}
            for k, v in results.items():
                if isinstance(v, dict):
                    serializable_results[k] = {
                        k2: str(v2)
                        if not isinstance(v2, (int, float, str, bool, list, dict))
                        else v2
                        for k2, v2 in v.items()
                    }
                else:
                    serializable_results[k] = (
                        str(v)
                        if not isinstance(v, (int, float, str, bool, list, dict))
                        else v
                    )

            json.dump(serializable_results, f, indent=2)

        logger.info(f"Training completed successfully. Results saved to {results_path}")

        # Print summary
        if "evaluation_metrics" in results:
            metrics = results["evaluation_metrics"]
            logger.info("Evaluation metrics:")
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    logger.info(f"  {k}: {v:.4f}")

        if "hub_url" in results:
            logger.info(f"Model pushed to HuggingFace Hub: {results['hub_url']}")

        return 0

    except Exception as e:
        logger.error(f"Error during automated training: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
