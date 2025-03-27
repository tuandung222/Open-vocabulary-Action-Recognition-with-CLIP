"""
Evaluation module for CLIP HAR Project.

This module contains tools for evaluating models, computing metrics,
and visualizing evaluation results.
"""

from CLIP_HAR_PROJECT.evaluation.evaluator import ClassificationEvaluator, Evaluator
from CLIP_HAR_PROJECT.evaluation.metrics import (
    compute_accuracy_per_class,
    compute_classification_report,
    compute_confusion_matrix,
    compute_metrics,
)
from CLIP_HAR_PROJECT.evaluation.visualization import (
    plot_accuracy_per_class,
    plot_confusion_matrix,
    plot_misclassifications,
    visualize_attention,
)

__all__ = [
    "compute_metrics",
    "compute_confusion_matrix",
    "compute_classification_report",
    "compute_accuracy_per_class",
    "plot_confusion_matrix",
    "plot_accuracy_per_class",
    "plot_misclassifications",
    "visualize_attention",
    "Evaluator",
    "ClassificationEvaluator",
]
