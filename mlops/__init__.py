"""
MLOps utilities for CLIP HAR Project, including model tracking,
version control, and CI/CD integration.
"""

from CLIP_HAR_PROJECT.mlops.dvc_utils import (
    add_data_to_dvc,
    get_tracked_data,
    push_data_to_remote,
    setup_dvc_repo,
)
from CLIP_HAR_PROJECT.mlops.tracking import (
    load_model_from_mlflow,
    log_confusion_matrix,
    log_metrics,
    log_model,
    log_model_params,
    setup_mlflow,
)
