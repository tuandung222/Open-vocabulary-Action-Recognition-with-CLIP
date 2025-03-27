"""
Evaluation module for CLIP HAR Project.

This module contains tools for evaluating models, computing metrics,
and visualizing evaluation results.
"""

from CLIP_HAR_PROJECT.evaluation.metrics import (
    compute_metrics,
    compute_confusion_matrix,
    compute_classification_report,
    compute_accuracy_per_class,
)

from CLIP_HAR_PROJECT.evaluation.visualization import (
    plot_confusion_matrix,
    plot_accuracy_per_class,
    plot_misclassifications,
    visualize_attention,
)

from CLIP_HAR_PROJECT.evaluation.evaluator import Evaluator, ClassificationEvaluator

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
