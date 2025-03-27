# DVC Guide: Tracking Hugging Face Datasets

This guide explains how to use Data Version Control (DVC) to track Hugging Face datasets in this project.

## Setup

DVC is already installed as part of the project requirements. If you need to install it separately:

```bash
pip install dvc
```

## Basic Workflow

### 1. Initialize DVC (already done)

```bash
dvc init
```

### 2. Download the Human Action Recognition Dataset

```bash
# Run the download stage
dvc repro download_dataset

# Alternatively, call the script directly
python scripts/download_dataset.py --dataset Bingsu/Human_Action_Recognition --output_dir data/har_dataset --version v1
```

This downloads the complete dataset and saves it to `data/har_dataset/v1`.

### 3. Track the Dataset with Git

```bash
# Add the dataset to DVC tracking
dvc add data/har_dataset/v1

# Add the .dvc file to Git
git add data/har_dataset/v1.dvc .gitignore
git commit -m "Add Human Action Recognition dataset v1"
```

### 4. Create Dataset Variants

Our DVC pipeline includes predefined dataset variants:

#### Active Movements Dataset (running, cycling, fighting, dancing)

```bash
dvc repro active_movements_dataset
```

#### Balanced Dataset (equal samples per class)

```bash
dvc repro balanced_dataset
```

#### Social Interactions Dataset (clapping, hugging, fighting)

```bash
dvc repro social_dataset
```

### 5. Create Your Own Dataset Variant

Use the `scripts/filter_dataset.py` script to create custom dataset variants:

```bash
# Filter to specific classes
python scripts/filter_dataset.py --input_dir data/har_dataset/v1 --output_dir data/har_dataset/v1_custom --include_classes 0 1 2

# Exclude certain classes
python scripts/filter_dataset.py --input_dir data/har_dataset/v1 --output_dir data/har_dataset/v1_no_fighting --exclude_classes 6

# Balance a specific split
python scripts/filter_dataset.py --input_dir data/har_dataset/v1 --output_dir data/har_dataset/v1_balanced_train --balance --split train
```

Then track it with DVC:

```bash
dvc add data/har_dataset/v1_custom
git add data/har_dataset/v1_custom.dvc
git commit -m "Add custom dataset variant"
```

## Running Experiments with Different Dataset Versions

### 1. Train Models on Different Dataset Variants

Use DVC to run training with different dataset versions:

```bash
# Train on full dataset
dvc repro train_full

# Train on active movements dataset
dvc repro train_active

# Train on balanced dataset
dvc repro train_balanced
```

### 2. Compare Model Results

```bash
dvc repro compare_models
```

This generates comparison charts and reports in `outputs/model_comparison/`.

## Remote Storage

### 1. Set Up Remote Storage

```bash
# Add an S3 remote
dvc remote add -d myremote s3://my-bucket/dvc-storage

# Add a Google Drive remote
dvc remote add -d gdrive gdrive://folder-id

# Add a local remote (for team sharing via network drive)
dvc remote add -d local /path/to/network/drive
```

### 2. Push and Pull Datasets

```bash
# Push all data to remote
dvc push

# Push specific dataset
dvc push data/har_dataset/v1.dvc

# Pull all data from remote
dvc pull

# Pull specific dataset
dvc pull data/har_dataset/v1.dvc
```

## CI/CD Integration

Add DVC commands to your CI/CD pipeline:

```yaml
# Example GitHub Actions step
- name: Set up DVC
  uses: iterative/setup-dvc@v1

- name: Pull dataset
  run: dvc pull data/har_dataset/v1.dvc

- name: Train model
  run: dvc repro train_full
```

## Class ID Reference for HAR Dataset

When filtering by class IDs:

- 0: calling
- 1: clapping
- 2: cycling
- 3: dancing
- 4: drinking
- 5: eating
- 6: fighting
- 7: hugging
- 8: laughing
- 9: listening_to_music
- 10: running
- 11: sitting
- 12: sleeping
- 13: texting
- 14: using_laptop

## Troubleshooting

### Dataset Not Found

If you see "Dataset not found" errors when pulling from remote:

```bash
dvc update -R data/har_dataset/
```

### Large Files Issues

For very large datasets, use:

```bash
dvc add --no-commit data/large_dataset
dvc commit data/large_dataset.dvc
```

### Fixing Broken DVC Files

If .dvc files become corrupted:

```bash
dvc remove data/har_dataset/v1.dvc
dvc add data/har_dataset/v1
```

## Examples for Common Tasks

### Creating a Subset for Quick Testing

```bash
python scripts/filter_dataset.py --input_dir data/har_dataset/v1 --output_dir data/har_dataset/v1_test_small --max_samples_per_class 10 --balance
```

### Extracting Only Phone-Related Activities

```bash
python scripts/filter_dataset.py --input_dir data/har_dataset/v1 --output_dir data/har_dataset/v1_phone --include_classes 0 13
```

### Working with a Custom DVC Pipeline

Create a custom pipeline stage in `dvc.yaml`:

```yaml
custom_dataset:
  cmd: python scripts/filter_dataset.py --input_dir data/har_dataset/v1 --output_dir data/har_dataset/v1_custom --include_classes 0 1 2 3
  deps:
    - scripts/filter_dataset.py
    - data/har_dataset/v1
  outs:
    - data/har_dataset/v1_custom:
        persist: true
  frozen: true
```

Then run:

```bash
dvc repro custom_dataset
``` 