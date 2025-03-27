import os
import argparse
import torch
import mlflow
from typing import Dict, Optional, Any

from CLIP_HAR_PROJECT.models.clip_model import CLIPLabelRetriever
from CLIP_HAR_PROJECT.configs import get_config
from CLIP_HAR_PROJECT.mlops.tracking import (
    setup_mlflow,
    log_model_params,
    log_metrics,
    log_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy CLIP HAR model to MLflow")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint",
    )
    parser.add_argument("--config_path", type=str, help="Path to the model config file")
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="clip_har_production",
        help="MLflow experiment name",
    )
    parser.add_argument("--run_name", type=str, default=None, help="MLflow run name")
    parser.add_argument(
        "--tracking_uri", type=str, default=None, help="MLflow tracking server URI"
    )
    parser.add_argument(
        "--register_model",
        action="store_true",
        help="Register model in MLflow Model Registry",
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Set model as production in MLflow Registry",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="clip_har_model",
        help="Name for registered model in MLflow",
    )
    parser.add_argument(
        "--artifact_path",
        type=str,
        default="model",
        help="Path within the run artifacts to store the model",
    )

    return parser.parse_args()


def load_model_and_metrics(model_path: str, config_path: Optional[str] = None) -> tuple:
    """
    Load a trained model and its metrics from checkpoint.

    Args:
        model_path: Path to the model checkpoint
        config_path: Optional path to model config file

    Returns:
        Tuple of (model, config, metrics)
    """
    print(f"Loading model from {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")

    # Extract metrics
    metrics = {}
    if "metrics" in checkpoint:
        metrics = checkpoint["metrics"]
    elif os.path.exists(os.path.join(os.path.dirname(model_path), "metrics.json")):
        import json

        with open(os.path.join(os.path.dirname(model_path), "metrics.json"), "r") as f:
            metrics = json.load(f)

    # Load config
    config = None
    if config_path:
        config = get_config(config_path)
    elif "config" in checkpoint:
        config = checkpoint["config"]

    # Initialize model
    if config:
        model = CLIPLabelRetriever(
            model_name=config.model.clip_model_name,
            num_classes=config.model.num_classes,
        )
    else:
        # Try to infer parameters if not provided
        model = CLIPLabelRetriever(
            model_name="openai/clip-vit-base-patch16",
            num_classes=15,  # Default for HAR dataset
        )

    # Load model weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model, config, metrics


def main():
    args = parse_args()

    # Set up MLflow
    experiment_id = setup_mlflow(
        experiment_name=args.experiment_name, tracking_uri=args.tracking_uri
    )

    # Start a new MLflow run
    with mlflow.start_run(run_name=args.run_name, experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        print(f"Started MLflow run: {run_id}")

        # Load model and metrics
        model, config, metrics = load_model_and_metrics(
            model_path=args.model_path, config_path=args.config_path
        )

        # Log model parameters
        if config:
            log_model_params(vars(config))
            print("Logged model parameters")

        # Log metrics if available
        if metrics:
            log_metrics(metrics)
            print("Logged metrics")

        # Log the model
        model_uri = log_model(
            model=model,
            artifact_path=args.artifact_path,
            metadata={
                "task": "human_action_recognition",
                "framework": "pytorch",
                "model_type": "clip",
                "num_classes": model.num_classes,
            },
        )
        print(f"Logged model: {model_uri}")

        # Register model if requested
        if args.register_model:
            registered_model = mlflow.register_model(
                model_uri=f"runs:/{run_id}/{args.artifact_path}", name=args.model_name
            )
            print(
                f"Registered model: {registered_model.name} (version {registered_model.version})"
            )

            # Set as production if requested
            if args.production:
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name=args.model_name,
                    version=registered_model.version,
                    stage="Production",
                )
                print(
                    f"Model {args.model_name} version {registered_model.version} set as Production"
                )

    print("Model deployment complete!")


if __name__ == "__main__":
    main()
