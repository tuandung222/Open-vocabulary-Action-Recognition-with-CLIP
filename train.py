#!/usr/bin/env python
# Main training script for HAR classification using CLIP

import argparse
import datetime
import os
import sys

import torch
import wandb
from transformers import CLIPImageProcessor, CLIPTokenizerFast

# Import project modules
from CLIP_HAR_PROJECT.configs.default import get_config
from CLIP_HAR_PROJECT.data.preprocessing import (
    get_class_mappings,
    prepare_har_dataset,
    visualize_samples,
)
from CLIP_HAR_PROJECT.models.clip_model import (
    CLIPLabelRetriever,
    freeze_clip_parameters,
    print_trainable_parameters,
)
from CLIP_HAR_PROJECT.training.trainer import DistributedTrainer, TrainingConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train CLIP model for HAR classification"
    )

    # Distributed training arguments
    parser.add_argument(
        "--distributed_mode",
        type=str,
        default="none",
        choices=["none", "ddp", "fsdp"],
        help="Distributed training mode",
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/clip-vit-base-patch16",
        help="CLIP model name or path",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="a photo of person/people who is/are {label}",
        help="Template for creating text prompts",
    )
    parser.add_argument(
        "--unfreeze_visual", action="store_true", help="Unfreeze visual encoder"
    )
    parser.add_argument(
        "--unfreeze_text", action="store_true", help="Unfreeze text encoder"
    )

    # Training arguments
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Training batch size (per GPU)"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="Evaluation batch size (per GPU)",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=None, help="Maximum number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--no_mixed_precision",
        action="store_true",
        help="Disable mixed precision training",
    )
    parser.add_argument(
        "--num_workers", type=int, default=None, help="Number of dataloader workers"
    )

    # Experiment tracking arguments
    parser.add_argument(
        "--no_tracking",
        action="store_true",
        help="Disable all tracking (MLflow and wandb)",
    )
    parser.add_argument(
        "--use_mlflow", action="store_true", help="Use MLflow for tracking"
    )
    parser.add_argument(
        "--no_mlflow", action="store_true", help="Disable MLflow tracking"
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use Weights & Biases for tracking"
    )
    parser.add_argument(
        "--no_wandb", action="store_true", help="Disable Weights & Biases tracking"
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None, help="Experiment name for tracking"
    )
    parser.add_argument(
        "--project_name", type=str, default=None, help="Project name for wandb"
    )
    parser.add_argument(
        "--group_name", type=str, default=None, help="Group name for wandb"
    )

    return parser.parse_args()


def update_config_from_args(config, args):
    """Update configuration from command line arguments."""
    # Model config
    if args.model_name:
        config.model.model_name = args.model_name
    if args.prompt_template:
        config.model.prompt_template = args.prompt_template
    config.model.unfreeze_visual_encoder = args.unfreeze_visual
    config.model.unfreeze_text_encoder = args.unfreeze_text

    # Training config
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.eval_batch_size:
        config.training.eval_batch_size = args.eval_batch_size
    if args.max_epochs:
        config.training.max_epochs = args.max_epochs
    if args.lr:
        config.training.lr = args.lr
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.no_mixed_precision:
        config.training.mixed_precision = False
    if args.num_workers is not None:
        config.training.num_workers = args.num_workers
    config.training.distributed_mode = args.distributed_mode

    # Logging config
    # Determine tracking configuration based on command line arguments
    use_mlflow = True
    use_wandb = True

    # Handle global tracking disable
    if args.no_tracking:
        use_mlflow = False
        use_wandb = False

    # Handle individual tracking options
    if args.use_mlflow:
        use_mlflow = True
    if args.no_mlflow:
        use_mlflow = False

    if args.use_wandb:
        use_wandb = True
    if args.no_wandb:
        use_wandb = False

    # Set tracking in config
    if hasattr(config, "tracking"):
        config.tracking.use_mlflow = use_mlflow
        config.tracking.use_wandb = use_wandb
    else:
        # Create tracking config section if not exists
        from types import SimpleNamespace

        config.tracking = SimpleNamespace(use_mlflow=use_mlflow, use_wandb=use_wandb)

    # Set wandb config if applicable
    if hasattr(config, "wandb"):
        if args.project_name:
            config.wandb.project_name = args.project_name
        if args.group_name:
            config.wandb.group_name = args.group_name
        if args.experiment_name:
            config.wandb.experiment_name = args.experiment_name
    else:
        # Create wandb config if not exists
        from types import SimpleNamespace

        config.wandb = SimpleNamespace(
            project_name=args.project_name or "clip_har",
            group_name=args.group_name,
            experiment_name=args.experiment_name,
        )

    # Set mlflow config if applicable
    if hasattr(config, "mlflow"):
        if args.experiment_name:
            config.mlflow.experiment_name = args.experiment_name
    else:
        # Create mlflow config if not exists
        from types import SimpleNamespace

        config.mlflow = SimpleNamespace(
            experiment_name=args.experiment_name or "clip_har",
            tracking_uri=None,
            artifacts_dir="./mlruns",
        )

    return config


def main():
    """Main training function."""
    # Parse command line arguments
    args = parse_args()

    # Get configuration
    config = get_config(args.distributed_mode)
    config = update_config_from_args(config, args)

    # Determine if this is the main process
    is_main_process = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )

    # Create the training pipeline
    from CLIP_HAR_PROJECT.pipeline.training_pipeline import TrainingPipeline

    pipeline = TrainingPipeline(
        distributed_mode=args.distributed_mode,
        use_mlflow=config.tracking.use_mlflow,
        use_wandb=config.tracking.use_wandb,
        experiment_name=args.experiment_name,
    )

    # Run the pipeline
    try:
        results = pipeline.run()

        # Print results summary (on main process only)
        if is_main_process:
            print("Training pipeline completed successfully!")
            if results.get("evaluation_metrics"):
                print(
                    f"Test accuracy: {results['evaluation_metrics'].get('accuracy', 0):.4f}"
                )
            if results.get("export_paths"):
                print(f"Exported models: {list(results['export_paths'].keys())}")
    except Exception as e:
        print(f"Error in training pipeline: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
