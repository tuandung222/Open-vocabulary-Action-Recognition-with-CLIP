#!/usr/bin/env python
# Script to download and save a Hugging Face dataset for DVC tracking

import argparse
import os
from pathlib import Path

from datasets import load_dataset


def main():
    """Download and save a Hugging Face dataset."""
    parser = argparse.ArgumentParser(
        description="Download and save a Hugging Face dataset for DVC tracking"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Bingsu/Human_Action_Recognition",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/har_dataset",
        help="Output directory for the dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to download (default: all splits)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Version tag for the dataset",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    if args.version:
        output_dir = output_dir / args.version
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading dataset: {args.dataset}")
    
    # Load the dataset
    if args.split:
        dataset = load_dataset(args.dataset, split=args.split)
        print(f"Downloaded split: {args.split}")
        
        # Save the dataset
        dataset.save_to_disk(str(output_dir))
    else:
        dataset = load_dataset(args.dataset)
        print(f"Downloaded all splits: {list(dataset.keys())}")
        
        # Save the dataset
        dataset.save_to_disk(str(output_dir))
    
    print(f"Dataset saved to: {output_dir}")
    
    # Create a metadata file with version information
    with open(output_dir / "metadata.txt", "w") as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Version: {args.version}\n")
        f.write(f"Splits: {args.split if args.split else 'all'}\n")
        f.write(f"Download date: {os.popen('date').read().strip()}\n")
        
        # Add dataset statistics
        if args.split:
            f.write(f"Number of examples: {len(dataset)}\n")
        else:
            for split, ds in dataset.items():
                f.write(f"Number of examples in {split}: {len(ds)}\n")
    
    print("Created metadata file")
    print("Done! You can now track this dataset with DVC:")
    print(f"  dvc add {output_dir}")


if __name__ == "__main__":
    main() 