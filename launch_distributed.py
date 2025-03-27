#!/usr/bin/env python
# Script to launch distributed training for HAR classification

import argparse
import os
import subprocess
import sys
from typing import List


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch distributed training for HAR classification"
    )

    # Distributed training arguments
    parser.add_argument(
        "--distributed_mode",
        type=str,
        default="ddp",
        choices=["ddp", "fsdp"],
        help="Distributed training mode",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: all available)",
    )
    parser.add_argument(
        "--master_addr", type=str, default="localhost", help="Master node address"
    )
    parser.add_argument(
        "--master_port", type=str, default="12355", help="Master node port"
    )

    # Training script arguments (passed through)
    parser.add_argument(
        "--model_name", type=str, default=None, help="CLIP model name or path"
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=None,
        help="Template for creating text prompts",
    )
    parser.add_argument(
        "--unfreeze_visual", action="store_true", help="Unfreeze visual encoder"
    )
    parser.add_argument(
        "--unfreeze_text", action="store_true", help="Unfreeze text encoder"
    )
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
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument(
        "--project_name", type=str, default=None, help="Project name for wandb"
    )
    parser.add_argument(
        "--group_name", type=str, default=None, help="Group name for wandb"
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None, help="Experiment name for wandb"
    )

    return parser.parse_args()


def get_num_gpus():
    """Get the number of available GPUs."""
    try:
        import torch

        return torch.cuda.device_count()
    except (ImportError, AttributeError):
        return 0


def build_torchrun_command(args) -> List[str]:
    """Build the torchrun command to launch distributed training."""
    # Determine number of GPUs
    num_gpus = args.num_gpus or get_num_gpus()
    if num_gpus <= 0:
        print("No GPUs available for distributed training")
        sys.exit(1)

    # Build base command
    cmd = [
        "torchrun",
        "--nproc_per_node",
        str(num_gpus),
        "--master_addr",
        args.master_addr,
        "--master_port",
        args.master_port,
        "CLIP_HAR_PROJECT/train.py",
        "--distributed_mode",
        args.distributed_mode,
    ]

    # Add training script arguments
    if args.model_name:
        cmd.extend(["--model_name", args.model_name])
    if args.prompt_template:
        cmd.extend(["--prompt_template", args.prompt_template])
    if args.unfreeze_visual:
        cmd.append("--unfreeze_visual")
    if args.unfreeze_text:
        cmd.append("--unfreeze_text")
    if args.batch_size is not None:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if args.eval_batch_size is not None:
        cmd.extend(["--eval_batch_size", str(args.eval_batch_size)])
    if args.max_epochs is not None:
        cmd.extend(["--max_epochs", str(args.max_epochs)])
    if args.lr is not None:
        cmd.extend(["--lr", str(args.lr)])
    if args.output_dir:
        cmd.extend(["--output_dir", args.output_dir])
    if args.no_mixed_precision:
        cmd.append("--no_mixed_precision")
    if args.num_workers is not None:
        cmd.extend(["--num_workers", str(args.num_workers)])
    if args.no_wandb:
        cmd.append("--no_wandb")
    if args.project_name:
        cmd.extend(["--project_name", args.project_name])
    if args.group_name:
        cmd.extend(["--group_name", args.group_name])
    if args.experiment_name:
        cmd.extend(["--experiment_name", args.experiment_name])

    return cmd


def main():
    """Main function to launch distributed training."""
    # Parse command line arguments
    args = parse_args()

    # Build torchrun command
    cmd = build_torchrun_command(args)

    # Print command
    print("Launching distributed training with command:")
    print(" ".join(cmd))

    # Set environment variables
    env = os.environ.copy()

    # Launch the process
    process = subprocess.Popen(cmd, env=env)

    try:
        # Wait for the process to complete
        process.wait()
        return process.returncode
    except KeyboardInterrupt:
        # Handle keyboard interrupt
        print("Received keyboard interrupt, terminating training...")
        process.terminate()
        process.wait()
        return 1


if __name__ == "__main__":
    sys.exit(main())
