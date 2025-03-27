"""
Evaluator module for CLIP HAR Project.

This module contains evaluator classes for different types of evaluation.
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from CLIP_HAR_PROJECT.evaluation.metrics import (
    compute_classification_report,
    compute_confusion_matrix,
    compute_metrics,
)
from CLIP_HAR_PROJECT.evaluation.visualization import (
    plot_accuracy_per_class,
    plot_confusion_matrix,
)

logger = logging.getLogger(__name__)


class BaseEvaluator:
    """Base class for all evaluators."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "evaluation_results",
        collate_fn: Optional[Callable] = None,
        visualize_results: bool = True,
    ):
        """
        Initialize the evaluator.

        Args:
            dataset: Dataset to evaluate on
            batch_size: Batch size for evaluation
            num_workers: Number of workers for data loading
            device: Device to use for evaluation
            output_dir: Directory to save evaluation results
            collate_fn: Collate function for the dataloader
            visualize_results: Whether to create visualizations
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.output_dir = output_dir
        self.collate_fn = collate_fn
        self.visualize_results = visualize_results

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Create dataloader
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

    def evaluate(self, model: torch.nn.Module) -> Dict[str, Any]:
        """
        Evaluate the model.

        Args:
            model: Model to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        raise NotImplementedError("Subclasses must implement this method")


class ClassificationEvaluator(BaseEvaluator):
    """Evaluator for classification tasks."""

    def __init__(
        self,
        dataset: Dataset,
        class_names: List[str],
        batch_size: int = 32,
        num_workers: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "evaluation_results",
        collate_fn: Optional[Callable] = None,
        visualize_results: bool = True,
        num_visualizations: int = 5,
    ):
        """
        Initialize the classification evaluator.

        Args:
            dataset: Dataset to evaluate on
            class_names: List of class names
            batch_size: Batch size for evaluation
            num_workers: Number of workers for data loading
            device: Device to use for evaluation
            output_dir: Directory to save evaluation results
            collate_fn: Collate function for the dataloader
            visualize_results: Whether to create visualizations
            num_visualizations: Number of examples to visualize
        """
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            output_dir=output_dir,
            collate_fn=collate_fn,
            visualize_results=visualize_results,
        )
        self.class_names = class_names
        self.num_visualizations = num_visualizations

    def evaluate(self, model: torch.nn.Module) -> Dict[str, Any]:
        """
        Evaluate the model on the classification task.

        Args:
            model: Model to evaluate

        Returns:
            Dictionary of evaluation metrics including:
            - accuracy, precision, recall, f1
            - confusion matrix
            - classification report
            - paths to visualization files
        """
        logger.info("Starting evaluation...")
        model.eval()
        model.to(self.device)

        # Initialize lists to store predictions and ground truth
        all_preds = []
        all_labels = []
        all_examples = []

        # Evaluate model
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Evaluating"):
                # Extract inputs and labels
                if isinstance(batch, dict):
                    inputs = batch["inputs"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    examples = batch.get(
                        "examples", None
                    )  # Original examples for visualization
                elif isinstance(batch, tuple) and len(batch) >= 2:
                    inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                    examples = batch[2] if len(batch) > 2 else None
                else:
                    raise ValueError("Batch format not supported")

                # Forward pass
                outputs = model(inputs)

                # Get predictions
                if isinstance(outputs, dict):
                    logits = outputs.get("logits", outputs.get("output", None))
                    if logits is None:
                        # Try to find the key for logits
                        for key, value in outputs.items():
                            if isinstance(value, torch.Tensor) and value.shape[
                                1
                            ] == len(self.class_names):
                                logits = value
                                break
                    if logits is None:
                        raise ValueError("Could not find logits in model outputs")
                elif isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    raise ValueError("Model output format not supported")

                # Get predictions
                preds = torch.argmax(logits, dim=1)

                # Store predictions and labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Store examples for visualization if available
                if examples is not None:
                    all_examples.extend(examples)

        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Compute metrics
        metrics = compute_metrics(all_labels, all_preds, average="weighted")
        confusion_mat = compute_confusion_matrix(
            all_labels, all_preds, len(self.class_names)
        )
        classification_rep = compute_classification_report(
            all_labels, all_preds, target_names=self.class_names
        )

        # Create results dictionary
        results = {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "confusion_matrix": confusion_mat,
            "classification_report": classification_rep,
            "predictions": all_preds,
            "references": all_labels,
        }

        # Create visualizations if requested
        if self.visualize_results:
            visualization_files = {}

            # Plot confusion matrix
            cm_plot_path = os.path.join(self.output_dir, "confusion_matrix.png")
            plot_confusion_matrix(
                confusion_mat, class_names=self.class_names, output_path=cm_plot_path
            )
            visualization_files["confusion_matrix"] = cm_plot_path

            # Plot accuracy per class
            acc_plot_path = os.path.join(self.output_dir, "accuracy_per_class.png")
            plot_accuracy_per_class(
                all_labels,
                all_preds,
                class_names=self.class_names,
                output_path=acc_plot_path,
            )
            visualization_files["accuracy_per_class"] = acc_plot_path

            # Add visualization files to results
            results["visualization_files"] = visualization_files

        return results


def get_evaluator(eval_type: str, dataset: Dataset, **kwargs) -> BaseEvaluator:
    """
    Factory function to get the appropriate evaluator.

    Args:
        eval_type: Type of evaluator to get ("classification", etc.)
        dataset: Dataset to evaluate on
        **kwargs: Additional arguments to pass to the evaluator

    Returns:
        An evaluator instance
    """
    if eval_type.lower() == "classification":
        return ClassificationEvaluator(dataset=dataset, **kwargs)
    else:
        raise ValueError(f"Evaluator type {eval_type} not supported")
