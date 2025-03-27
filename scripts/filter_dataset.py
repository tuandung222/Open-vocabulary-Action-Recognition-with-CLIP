#!/usr/bin/env python
# Script to filter a downloaded Hugging Face dataset for DVC tracking

import argparse
import os
from pathlib import Path
import json

from datasets import load_from_disk


def filter_by_classes(dataset, include_classes=None, exclude_classes=None):
    """Filter dataset to only include or exclude specific classes."""
    if include_classes:
        return dataset.filter(lambda x: x['labels'] in include_classes)
    elif exclude_classes:
        return dataset.filter(lambda x: x['labels'] not in exclude_classes)
    return dataset


def balance_dataset(dataset, max_samples_per_class=None):
    """Balance dataset to have equal samples per class."""
    # Count samples per class
    class_counts = {}
    for example in dataset:
        label = example['labels']
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    
    # Determine target count
    if max_samples_per_class:
        target_count = min(max_samples_per_class, min(class_counts.values()))
    else:
        target_count = min(class_counts.values())
    
    # Filter to balance classes
    balanced_examples = []
    class_counters = {label: 0 for label in class_counts}
    
    for example in dataset:
        label = example['labels']
        if class_counters[label] < target_count:
            balanced_examples.append(example)
            class_counters[label] += 1
    
    return dataset.select(range(len(balanced_examples)))


def main():
    """Filter a Hugging Face dataset."""
    parser = argparse.ArgumentParser(
        description="Filter a downloaded Hugging Face dataset for DVC tracking"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory with the downloaded dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for the filtered dataset",
    )
    parser.add_argument(
        "--include_classes",
        type=int,
        nargs="+",
        default=None,
        help="List of class IDs to include (e.g., 0 1 2)",
    )
    parser.add_argument(
        "--exclude_classes",
        type=int,
        nargs="+",
        default=None,
        help="List of class IDs to exclude (e.g., 5 6 7)",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Balance classes to have equal samples",
    )
    parser.add_argument(
        "--max_samples_per_class",
        type=int,
        default=None,
        help="Maximum samples per class (for balancing)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to filter (default: all splits)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading dataset from: {args.input_dir}")
    
    # Load the dataset
    if args.split:
        dataset = load_from_disk(args.input_dir)[args.split]
        print(f"Loaded split: {args.split}")
        
        # Apply filters
        original_size = len(dataset)
        print(f"Original dataset size: {original_size}")
        
        if args.include_classes:
            dataset = filter_by_classes(dataset, include_classes=args.include_classes)
            print(f"Filtered to include classes {args.include_classes}")
        
        if args.exclude_classes:
            dataset = filter_by_classes(dataset, exclude_classes=args.exclude_classes)
            print(f"Filtered to exclude classes {args.exclude_classes}")
        
        if args.balance:
            dataset = balance_dataset(dataset, args.max_samples_per_class)
            print(f"Balanced classes" + 
                  (f" with max {args.max_samples_per_class} samples per class" if args.max_samples_per_class else ""))
        
        # Save the filtered dataset
        dataset.save_to_disk(str(output_dir))
        print(f"Filtered dataset size: {len(dataset)}")
        
    else:
        dataset = load_from_disk(args.input_dir)
        print(f"Loaded all splits: {list(dataset.keys())}")
        
        # Process each split
        for split in dataset:
            split_dataset = dataset[split]
            original_size = len(split_dataset)
            print(f"Original {split} dataset size: {original_size}")
            
            if args.include_classes:
                split_dataset = filter_by_classes(split_dataset, include_classes=args.include_classes)
                print(f"Filtered {split} to include classes {args.include_classes}")
            
            if args.exclude_classes:
                split_dataset = filter_by_classes(split_dataset, exclude_classes=args.exclude_classes)
                print(f"Filtered {split} to exclude classes {args.exclude_classes}")
            
            if args.balance:
                split_dataset = balance_dataset(split_dataset, args.max_samples_per_class)
                print(f"Balanced {split} classes" + 
                      (f" with max {args.max_samples_per_class} samples per class" if args.max_samples_per_class else ""))
            
            # Update the split in the dataset
            dataset[split] = split_dataset
            print(f"Filtered {split} dataset size: {len(split_dataset)}")
        
        # Save the filtered dataset
        dataset.save_to_disk(str(output_dir))
    
    print(f"Dataset saved to: {output_dir}")
    
    # Create a metadata file with filtering information
    filter_info = {
        "input_dir": args.input_dir,
        "include_classes": args.include_classes,
        "exclude_classes": args.exclude_classes,
        "balanced": args.balance,
        "max_samples_per_class": args.max_samples_per_class,
        "split": args.split,
    }
    
    with open(output_dir / "filter_metadata.json", "w") as f:
        json.dump(filter_info, f, indent=2)
    
    print("Created metadata file with filtering information")
    print("Done! You can now track this filtered dataset with DVC:")
    print(f"  dvc add {output_dir}")


if __name__ == "__main__":
    main() 