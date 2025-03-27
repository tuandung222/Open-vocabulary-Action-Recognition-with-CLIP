"""
MLOps utilities for CLIP HAR Project, including model tracking, 
version control, and CI/CD integration.
"""

from CLIP_HAR_PROJECT.mlops.tracking import (
    setup_mlflow,
    log_model_params,
    log_metrics,
    log_model,
    log_confusion_matrix,
    load_model_from_mlflow,
)

from CLIP_HAR_PROJECT.mlops.dvc_utils import (
    setup_dvc_repo,
    add_data_to_dvc,
    get_tracked_data,
    push_data_to_remote,
)
