import torch
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from transformers import CLIPTokenizerFast, CLIPImageProcessor
from functools import partial
from typing import Dict, List, Tuple, Optional, Callable, Union, Any


def load_har_dataset(dataset_name="Bingsu/Human_Action_Recognition"):
    """
    Load the HAR dataset from HuggingFace.

    Args:
        dataset_name: The name of the dataset on HuggingFace

    Returns:
        The loaded dataset
    """
    return load_dataset(dataset_name)


def get_class_mappings(dataset):
    """
    Get class mappings from the dataset.

    Args:
        dataset: The HAR dataset

    Returns:
        Tuple of (string_to_id, id_to_string, class_names)
    """
    # Predefined mappings for HAR dataset
    string_to_id = {
        "calling": 0,
        "clapping": 1,
        "cycling": 2,
        "dancing": 3,
        "drinking": 4,
        "eating": 5,
        "fighting": 6,
        "hugging": 7,
        "laughing": 8,
        "listening_to_music": 9,
        "running": 10,
        "sitting": 11,
        "sleeping": 12,
        "texting": 13,
        "using_laptop": 14,
    }
    id_to_string = {v: k for k, v in string_to_id.items()}
    class_names = list(string_to_id.keys())

    return string_to_id, id_to_string, class_names


def split_dataset(dataset, val_ratio=0.15, test_ratio=0.25, seed=42):
    """
    Split the dataset into train, validation, and test sets.

    Args:
        dataset: The dataset to split
        val_ratio: The ratio of validation set
        test_ratio: The ratio of test set
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # First split: train vs. (val+test)
    first_split_dataset = dataset["train"].train_test_split(
        test_size=val_ratio + test_ratio, seed=seed, stratify_by_column="labels"
    )

    # Second split: val vs. test
    second_split_dataset = first_split_dataset["test"].train_test_split(
        test_size=test_ratio / (test_ratio + val_ratio),
        seed=seed,
        stratify_by_column="labels",
    )

    train_dataset = first_split_dataset["train"]
    val_dataset = second_split_dataset["train"]
    test_dataset = second_split_dataset["test"]

    return train_dataset, val_dataset, test_dataset


def transform_class_labels(
    items,
    tokenizer,
    prompt_template="a photo of person/people who is/are {label}",
    id2label=None,
    label_column_name="labels",
):
    """
    Transform class labels to text prompts and tokenize them.

    Args:
        items: Batch of dataset items
        tokenizer: CLIP tokenizer
        prompt_template: Template for creating prompts from labels
        id2label: Mapping from label IDs to label strings
        label_column_name: Name of the label column in the dataset

    Returns:
        Transformed items
    """
    assert label_column_name in items

    # Convert label IDs to strings if id2label is provided
    if id2label:
        labels = [id2label[x] for x in items[label_column_name]]
    else:
        labels = items[label_column_name]

    # Create prompts
    label_prompts = [prompt_template.format(label=label) for label in labels]

    # Tokenize prompts
    output = tokenizer(label_prompts, padding=True, return_tensors="pt")

    # Add tokenized prompts to items
    items["input_ids"] = output["input_ids"]
    items["attention_mask"] = output["attention_mask"]
    items["label_id"] = items[label_column_name]

    return items


def transform_image(items, image_processor):
    """
    Transform images using the CLIP image processor.

    Args:
        items: Batch of dataset items
        image_processor: CLIP image processor

    Returns:
        Transformed items
    """
    assert "image" in items

    # Process images
    output = image_processor(items["image"], return_tensors="pt")

    # Add processed images to items
    items["pixel_values"] = output["pixel_values"]

    return items


def transform_dataset(
    dataset,
    tokenizer,
    image_processor,
    id2label=None,
    prompt_template="a photo of person/people who is/are {label}",
    remove_columns=["labels"],
):
    """
    Transform a dataset for use with CLIP.

    Args:
        dataset: The dataset to transform
        tokenizer: CLIP tokenizer
        image_processor: CLIP image processor
        id2label: Mapping from label IDs to label strings
        prompt_template: Template for creating prompts from labels
        remove_columns: Columns to remove from the dataset

    Returns:
        Transformed dataset
    """
    # Apply label transformation
    transform_labels_fn = partial(
        transform_class_labels,
        tokenizer=tokenizer,
        id2label=id2label,
        prompt_template=prompt_template,
    )

    transformed_dataset = dataset.map(
        transform_labels_fn,
        batched=True,
        remove_columns=remove_columns if remove_columns else [],
    )

    # Set transform for images
    transformed_dataset.set_transform(
        partial(transform_image, image_processor=image_processor)
    )

    return transformed_dataset


def prepare_har_dataset(
    tokenizer, image_processor, val_ratio=0.15, test_ratio=0.25, seed=42
):
    """
    Prepare the HAR dataset for training with CLIP.

    Args:
        tokenizer: CLIP tokenizer
        image_processor: CLIP image processor
        val_ratio: The ratio of validation set
        test_ratio: The ratio of test set
        seed: Random seed for reproducibility

    Returns:
        Dictionary of transformed datasets
    """
    # Load dataset
    dataset = load_har_dataset()

    # Get class mappings
    string_to_id, id_to_string, class_names = get_class_mappings(dataset)

    # Split dataset
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )

    # Transform datasets
    train_dataset = transform_dataset(
        train_dataset, tokenizer, image_processor, id_to_string
    )
    val_dataset = transform_dataset(
        val_dataset, tokenizer, image_processor, id_to_string
    )
    test_dataset = transform_dataset(
        test_dataset, tokenizer, image_processor, id_to_string
    )

    # Create dataset dictionary
    final_dataset = DatasetDict(
        {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    )

    return final_dataset, class_names


def collate_fn(items):
    """
    Collate function for DataLoader.

    Args:
        items: List of dataset items

    Returns:
        Batch dictionary
    """
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in items]),
        "input_ids": torch.tensor([item["input_ids"] for item in items]),
        "attention_mask": torch.tensor([item["attention_mask"] for item in items]),
        "label_id": torch.tensor([item["label_id"] for item in items]),
    }


def visualize_samples(dataset, id2string, rows=3, cols=5):
    """
    Visualize samples from the dataset.

    Args:
        dataset: The dataset
        id2string: Mapping from label IDs to label strings
        rows: Number of rows
        cols: Number of columns
    """
    import matplotlib.pyplot as plt

    samples = dataset.shuffle().select(range(rows * cols))
    fig = plt.figure(figsize=(cols * 4, rows * 4))

    for i in range(rows * cols):
        img = samples[i]["image"]
        label = samples[i]["labels"]  # ID
        label_str = id2string[label]

        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(img)
        ax.set_title(label_str)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def create_class_distribution_visualizations(dataset):
    """
    Create visualizations for class distribution.

    Args:
        dataset: The dataset
    """
    import matplotlib.pyplot as plt
    import plotly.express as px
    import pandas as pd

    # Count classes
    class_count = {}
    for label in dataset["labels"]:
        if label not in class_count:
            class_count[label] = 0
        class_count[label] += 1

    # Calculate class distribution
    class_dist = {k: v / len(dataset) for k, v in class_count.items()}

    # Get class names
    _, id2string, class_names = get_class_mappings(None)

    # Create pie chart
    fig = px.pie(
        names=[id2string[id] for id in class_dist.keys()],
        values=list(class_dist.values()),
        width=600,
    )
    fig.update_layout({"title": {"text": "Class Distribution", "x": 0.5}})

    # Create bar chart
    fig2 = px.bar(
        x=[id2string[id] for id in class_dist.keys()],
        y=list(class_dist.values()),
        title="Class Frequency",
    )
    fig2.update_layout({"title": {"x": 0.1}})

    return fig, fig2
