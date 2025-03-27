#!/usr/bin/env python
"""
Quick training script that runs 5 epochs and pushes to HuggingFace Hub.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add the project root to sys.path to enable imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from CLIP_HAR_PROJECT.configs import get_quick_test_config
from CLIP_HAR_PROJECT.mlops.automated_training import run_automated_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a quick 5-epoch training and push to HuggingFace Hub"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/quick_test",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--hub_repo_id",
        type=str,
        default="tuandunghcmut/temp_push",
        help="HuggingFace Hub repository ID",
    )
    parser.add_argument(
        "--private_repo",
        action="store_true",
        help="Make the HuggingFace Hub repository private",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="quick_test",
        help="Name for the experiment",
    )
    parser.add_argument(
        "--distributed_mode",
        type=str,
        default="none",
        choices=["none", "ddp", "fsdp"],
        help="Training mode (none, ddp, fsdp)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get the quick test config
    config = get_quick_test_config()
    
    # Create a config file path
    config_path = os.path.join(args.output_dir, "quick_test_config.yaml")
    
    # Using the config directly without saving to file
    try:
        logger.info(f"Starting quick training with {config.training.max_epochs} epochs")
        logger.info(f"Will push to HuggingFace Hub at {args.hub_repo_id}")
        
        # Run the automated training
        results = run_automated_training(
            config_path=None,  # We'll pass the config directly
            output_dir=args.output_dir,
            push_to_hub=True,
            hub_repo_id=args.hub_repo_id,
            private_repo=args.private_repo,
            experiment_name=args.experiment_name,
            distributed_mode=args.distributed_mode,
            config=config,  # Pass the config directly
        )
        
        logger.info("Training completed successfully")
        
        if "hub_url" in results:
            logger.info(f"Model pushed to HuggingFace Hub: {results['hub_url']}")
        
        return 0
    except Exception as e:
        logger.error(f"Error during quick training: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main()) 