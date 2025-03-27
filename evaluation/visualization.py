"""
Visualization functions for CLIP HAR evaluation.

This module contains functions for visualizing evaluation results,
including confusion matrices and accuracy charts.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Optional[str] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    figsize: tuple = (10, 8),
    cmap: str = "Blues",
) -> plt.Figure:
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        output_path: Path to save the figure
        normalize: Whether to normalize the confusion matrix
        title: Title of the plot
        figsize: Figure size
        cmap: Colormap

    Returns:
        Matplotlib figure
    """
    # Normalize the confusion matrix if requested
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # Replace NaN with 0
        fmt = ".2f"
    else:
        fmt = "d"

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    # Set labels and title
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(title)

    # Rotate x tick labels
    plt.xticks(rotation=45, ha="right")

    # Adjust layout
    plt.tight_layout()

    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_accuracy_per_class(
    references: np.ndarray,
    predictions: np.ndarray,
    class_names: List[str],
    output_path: Optional[str] = None,
    title: str = "Accuracy per Class",
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """
    Plot accuracy for each class.

    Args:
        references: Ground truth labels
        predictions: Predicted labels
        class_names: List of class names
        output_path: Path to save the figure
        title: Title of the plot
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Compute confusion matrix
    cm = confusion_matrix(references, predictions, labels=range(len(class_names)))

    # Compute per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    per_class_acc = np.nan_to_num(per_class_acc)  # Replace NaN with 0

    # Sort classes by accuracy (descending)
    sorted_indices = np.argsort(per_class_acc)[::-1]
    sorted_acc = per_class_acc[sorted_indices]
    sorted_classes = [class_names[i] for i in sorted_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot bar chart
    sns.barplot(x=sorted_classes, y=sorted_acc, ax=ax)

    # Set labels and title
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Class")
    ax.set_title(title)

    # Rotate x tick labels
    plt.xticks(rotation=45, ha="right")

    # Add accuracy values as text
    for i, v in enumerate(sorted_acc):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom")

    # Add overall accuracy line
    overall_acc = np.mean(per_class_acc)
    ax.axhline(
        y=overall_acc,
        color="r",
        linestyle="--",
        label=f"Overall Accuracy: {overall_acc:.3f}",
    )
    ax.legend()

    # Set y-axis limits
    ax.set_ylim(0, 1.1)

    # Adjust layout
    plt.tight_layout()

    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_misclassifications(
    dataset: Any,
    predictions: np.ndarray,
    references: np.ndarray,
    class_names: List[str],
    output_dir: str,
    num_samples: int = 10,
    filename: str = "misclassifications.png",
    figsize: Optional[Tuple[int, int]] = None,
    dpi: int = 300,
) -> str:
    """
    Plot and save examples of misclassified samples.

    Args:
        dataset: Dataset containing samples
        predictions: Model predictions (class indices)
        references: Ground truth labels (class indices)
        class_names: List of class names
        output_dir: Directory to save the plot
        num_samples: Number of misclassified samples to visualize
        filename: Filename for the saved plot
        figsize: Figure size (width, height)
        dpi: Resolution of the saved image

    Returns:
        Path to the saved plot
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find misclassified samples
    misclassified_indices = np.where(predictions != references)[0]

    if len(misclassified_indices) == 0:
        print("No misclassified samples found.")
        return None

    # Limit number of samples
    num_samples = min(num_samples, len(misclassified_indices))
    selected_indices = np.random.choice(
        misclassified_indices, num_samples, replace=False
    )

    # Calculate grid dimensions
    cols = min(5, num_samples)
    rows = (num_samples + cols - 1) // cols

    # Set figure size
    if figsize is None:
        figsize = (cols * 4, rows * 4)

    # Create figure and plots
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if rows > 1 or cols > 1 else [axes]

    # Process each sample
    for i, idx in enumerate(selected_indices):
        if i >= len(axes):
            break

        # Get sample
        sample = dataset[idx]
        image = sample["image"] if "image" in sample else sample["pixel_values"]

        # Get true and predicted labels
        true_idx = references[idx]
        pred_idx = predictions[idx]
        true_label = class_names[true_idx]
        pred_label = class_names[pred_idx]

        # Display image
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] == 3:
                # Convert CHW -> HWC for display
                img_display = image.permute(1, 2, 0).cpu().numpy()

                # Normalize if needed
                if img_display.max() <= 1.0:
                    img_display = (img_display * 255).astype(np.uint8)
            else:
                img_display = image.cpu().numpy()
        elif isinstance(image, np.ndarray):
            img_display = image
        elif isinstance(image, Image.Image):
            img_display = np.array(image)
        else:
            continue  # Skip if image format not supported

        # Plot image
        axes[i].imshow(img_display)

        # Add title with true and predicted labels
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=10)
        axes[i].axis("off")

    # Hide unused subplots
    for i in range(len(selected_indices), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    # Save plot
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=dpi)
    plt.close()

    return output_path


def visualize_attention(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    output_dir: str,
    filename: str = "attention_map.png",
    figsize: Tuple[int, int] = (10, 10),
    dpi: int = 300,
) -> str:
    """
    Visualize attention maps for a given image.

    Args:
        model: Model with attention visualization capability
        image_tensor: Input image tensor
        output_dir: Directory to save the plot
        filename: Filename for the saved plot
        figsize: Figure size (width, height)
        dpi: Resolution of the saved image

    Returns:
        Path to the saved plot
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Ensure model is in eval mode
    model.eval()

    # Move inputs to the same device as model
    device = next(model.parameters()).device
    if isinstance(image_tensor, torch.Tensor):
        image_tensor = image_tensor.to(device)

    # Get attention maps (implementation depends on model architecture)
    if hasattr(model, "get_attention_maps"):
        # If model has a dedicated method for attention maps
        attention_map = model.get_attention_maps(image_tensor)
    else:
        # Fallback: Extract from model's visual encoder (CLIP-specific)
        try:
            # For CLIP models
            if hasattr(model, "visual_encoder"):
                visual_encoder = model.visual_encoder

                # Requires model modification to return attention maps
                # This is a placeholder - actual implementation will depend on model internals
                attention_map = (
                    None  # visual_encoder.get_last_selfattention(image_tensor)
                )

                if attention_map is None:
                    print("Attention map visualization not implemented for this model.")
                    return None
            else:
                print("Model doesn't support attention visualization.")
                return None
        except Exception as e:
            print(f"Error extracting attention maps: {e}")
            return None

    # Convert image tensor for display
    if isinstance(image_tensor, torch.Tensor):
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]  # Take first image if batch

        # Convert CHW -> HWC for display
        img_display = image_tensor.permute(1, 2, 0).cpu().numpy()

        # Normalize if needed
        if img_display.max() <= 1.0:
            img_display = (img_display * 255).astype(np.uint8)

    # Create figure
    plt.figure(figsize=figsize)

    # Plot original image and attention maps
    num_attention_heads = attention_map.shape[0]
    rows = int(np.ceil(np.sqrt(num_attention_heads + 1)))
    cols = int(np.ceil((num_attention_heads + 1) / rows))

    # Plot original image
    plt.subplot(rows, cols, 1)
    plt.imshow(img_display)
    plt.title("Original Image")
    plt.axis("off")

    # Plot attention maps
    for i in range(num_attention_heads):
        plt.subplot(rows, cols, i + 2)
        plt.imshow(attention_map[i], cmap="viridis")
        plt.title(f"Attention Head {i+1}")
        plt.axis("off")

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=dpi)
    plt.close()

    return output_path


def plot_metric_comparison(
    metrics_df: pd.DataFrame,
    output_dir: str,
    x_col: str = "Model",
    y_col: str = "accuracy",
    title: str = "Model Comparison",
    filename: str = "model_comparison.png",
    figsize: Tuple[int, int] = (10, 6),
    color: str = "skyblue",
    dpi: int = 300,
) -> str:
    """
    Plot and save comparison of models based on a metric.

    Args:
        metrics_df: DataFrame with model metrics
        output_dir: Directory to save the plot
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        title: Plot title
        filename: Filename for the saved plot
        figsize: Figure size (width, height)
        color: Bar color
        dpi: Resolution of the saved image

    Returns:
        Path to the saved plot
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create figure
    plt.figure(figsize=figsize)

    # Sort DataFrame by metric
    sorted_df = metrics_df.sort_values(y_col, ascending=False)

    # Plot bars
    bars = plt.bar(sorted_df[x_col], sorted_df[y_col], color=color)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.005,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            rotation=0,
        )

    # Customize plot
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=dpi)
    plt.close()

    return output_path


def create_evaluation_report(
    metrics: Dict[str, Any], output_dir: str, title: str = "Evaluation Report"
) -> Dict[str, str]:
    """
    Create a comprehensive evaluation report with
    visualizations and metrics.

    Args:
        metrics: Dictionary of evaluation metrics
        output_dir: Directory to save output files
        title: Title of the report

    Returns:
        Dictionary mapping visualization names to file paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize dictionary to store visualization paths
    visualization_files = {}

    # Plot confusion matrix if available
    if "confusion_matrix" in metrics:
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            class_names=metrics.get("class_names", []),
            output_path=cm_path,
            title="Confusion Matrix",
        )
        visualization_files["confusion_matrix"] = cm_path

    # Plot per-class accuracy if available
    if "predictions" in metrics and "references" in metrics:
        acc_path = os.path.join(output_dir, "accuracy_per_class.png")
        plot_accuracy_per_class(
            metrics["references"],
            metrics["predictions"],
            class_names=metrics.get("class_names", []),
            output_path=acc_path,
            title="Accuracy per Class",
        )
        visualization_files["accuracy_per_class"] = acc_path

    return visualization_files
