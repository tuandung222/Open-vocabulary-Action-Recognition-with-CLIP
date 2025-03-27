"""
Metrics for evaluating CLIP HAR models.

This module contains functions for computing evaluation metrics for
classification tasks in human action recognition.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report


def compute_metrics(
    references: np.ndarray, predictions: np.ndarray, average: str = "weighted"
) -> Dict[str, float]:
    """
    Compute basic classification metrics.

    Args:
        references: Ground truth labels
        predictions: Predicted labels
        average: Averaging method for precision, recall, and f1

    Returns:
        Dictionary of metrics
    """
    return {
        "accuracy": float(accuracy_score(references, predictions)),
        "precision": float(
            precision_score(references, predictions, average=average, zero_division=0)
        ),
        "recall": float(
            recall_score(references, predictions, average=average, zero_division=0)
        ),
        "f1": float(
            f1_score(references, predictions, average=average, zero_division=0)
        ),
    }


def compute_confusion_matrix(
    references: np.ndarray, predictions: np.ndarray, num_classes: int
) -> np.ndarray:
    """
    Compute the confusion matrix.

    Args:
        references: Ground truth labels
        predictions: Predicted labels
        num_classes: Number of classes

    Returns:
        Confusion matrix
    """
    return confusion_matrix(references, predictions, labels=range(num_classes))


def compute_classification_report(
    references: np.ndarray,
    predictions: np.ndarray,
    target_names: Optional[List[str]] = None,
    output_dict: bool = True,
) -> Dict[str, Any]:
    """
    Compute a detailed classification report.

    Args:
        references: Ground truth labels
        predictions: Predicted labels
        target_names: List of class names
        output_dict: Whether to return a dict (True) or string (False)

    Returns:
        Classification report
    """
    return classification_report(
        references,
        predictions,
        target_names=target_names,
        output_dict=output_dict,
        zero_division=0,
    )


def compute_accuracy_per_class(
    references: np.ndarray, predictions: np.ndarray, num_classes: int
) -> np.ndarray:
    """
    Compute accuracy for each class.

    Args:
        references: Ground truth labels
        predictions: Predicted labels
        num_classes: Number of classes

    Returns:
        Array of per-class accuracy values
    """
    conf_mat = compute_confusion_matrix(references, predictions, num_classes)
    return conf_mat.diagonal() / conf_mat.sum(axis=1)
