import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

# Import wandb conditionally
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Abstract base class for experiment tracking platforms.
    """

    def __init__(self, experiment_name: str):
        """Initialize tracker with experiment name."""
        self.experiment_name = experiment_name

    def start_run(self, run_name: Optional[str] = None) -> Any:
        """Start a new run."""
        raise NotImplementedError

    def end_run(self) -> None:
        """End the current run."""
        raise NotImplementedError

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters."""
        raise NotImplementedError

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log metrics."""
        raise NotImplementedError

    def log_artifact(
        self, local_path: str, artifact_path: Optional[str] = None
    ) -> None:
        """Log an artifact."""
        raise NotImplementedError

    def log_model(self, model: torch.nn.Module, **kwargs) -> Any:
        """Log a PyTorch model."""
        raise NotImplementedError

    def log_figure(self, figure: plt.Figure, filename: str) -> None:
        """Log a matplotlib figure."""
        raise NotImplementedError


class MLflowTracker(ExperimentTracker):
    """
    MLflow implementation of the ExperimentTracker interface.
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        artifacts_dir: str = "./mlruns",
    ):
        """
        Initialize MLflow tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (None for local)
            artifacts_dir: Local directory to store artifacts
        """
        super().__init__(experiment_name)

        # Set tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            os.makedirs(artifacts_dir, exist_ok=True)
            mlflow.set_tracking_uri(f"file:{artifacts_dir}")

        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=os.path.join(artifacts_dir, experiment_name),
            )
        else:
            self.experiment_id = experiment.experiment_id

        mlflow.set_experiment(experiment_name)
        self.active_run = None

    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """
        Start a new MLflow run.

        Args:
            run_name: Name for the run

        Returns:
            Active MLflow run
        """
        self.active_run = mlflow.start_run(run_name=run_name)
        return self.active_run

    def end_run(self) -> None:
        """End the current MLflow run."""
        if self.active_run:
            mlflow.end_run()
            self.active_run = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to MLflow.

        Args:
            params: Dictionary of parameters to log
        """
        # Handle nested dictionaries by flattening them
        flattened_params = {}

        def flatten_dict(d, parent_key=""):
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    flatten_dict(v, new_key)
                else:
                    # Skip complex objects that MLflow can't serialize
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        flattened_params[new_key] = v

        flatten_dict(params)

        # Log flattened parameters
        mlflow.log_params(flattened_params)

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """
        Log metrics to MLflow.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for the metrics
        """
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(
        self, local_path: str, artifact_path: Optional[str] = None
    ) -> None:
        """
        Log an artifact to MLflow.

        Args:
            local_path: Path to the artifact
            artifact_path: Path within the artifact directory
        """
        mlflow.log_artifact(local_path, artifact_path)

    def log_model(
        self,
        model: torch.nn.Module,
        artifact_path: str = "model",
        conda_env: Optional[Dict[str, Any]] = None,
        code_paths: Optional[List[str]] = None,
        registered_model_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Log a PyTorch model to MLflow.

        Args:
            model: PyTorch model to log
            artifact_path: Path within run artifacts to save the model
            conda_env: Dictionary that describes conda environment
            code_paths: List of local filesystem paths to Python file dependencies
            registered_model_name: Name to register the model under
            metadata: Dictionary of metadata to save with model

        Returns:
            URI of the logged model
        """
        if conda_env is None:
            conda_env = {
                "name": "clip_har_env",
                "channels": ["defaults", "pytorch", "conda-forge"],
                "dependencies": [
                    "python=3.8",
                    "pip",
                    {
                        "pip": [
                            "torch>=2.0.0",
                            "torchvision>=0.15.0",
                            "transformers>=4.30.0",
                            "mlflow>=2.3.0",
                        ]
                    },
                ],
            }

        model_info = mlflow.pytorch.log_model(
            model,
            artifact_path=artifact_path,
            conda_env=conda_env,
            code_paths=code_paths,
            registered_model_name=registered_model_name,
            metadata=metadata,
        )

        return model_info.model_uri

    def log_figure(self, figure: plt.Figure, filename: str) -> None:
        """
        Log a matplotlib figure to MLflow.

        Args:
            figure: Matplotlib figure to log
            filename: Filename for the figure
        """
        # Save figure to a temporary file
        figure.savefig(filename)

        # Log the file as an artifact
        self.log_artifact(filename)

        # Remove the temporary file
        os.remove(filename)

    def log_confusion_matrix(
        self,
        y_true: List[int],
        y_pred: List[int],
        class_names: List[str],
        normalize: bool = True,
        title: str = "Confusion Matrix",
        figure_size: tuple = (10, 8),
    ) -> None:
        """
        Log a confusion matrix visualization to MLflow.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            class_names: List of class names
            normalize: Whether to normalize confusion matrix values
            title: Title for the confusion matrix plot
            figure_size: Size of the figure
        """
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Create figure and plot
        plt.figure(figsize=figure_size)
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(title)
        plt.tight_layout()

        # Save and log the figure
        confusion_matrix_path = "confusion_matrix.png"
        plt.savefig(confusion_matrix_path)
        plt.close()

        self.log_artifact(confusion_matrix_path)
        os.remove(confusion_matrix_path)


class WandbTracker(ExperimentTracker):
    """
    Weights & Biases implementation of the ExperimentTracker interface.
    """

    def __init__(
        self,
        experiment_name: str,
        project_name: str = "clip_har",
        group: Optional[str] = None,
        entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize wandb tracker.

        Args:
            experiment_name: Name of the experiment (run name in wandb)
            project_name: Name of the wandb project
            group: Group name for the run
            entity: Team name or username
            config: Configuration dictionary for the run
        """
        super().__init__(experiment_name)

        if not WANDB_AVAILABLE:
            raise ImportError(
                "wandb is not installed. Install it with `pip install wandb`"
            )

        self.project_name = project_name
        self.group = group
        self.entity = entity
        self.config = config
        self.run = None

    def start_run(self, run_name: Optional[str] = None) -> "wandb.sdk.wandb_run.Run":
        """
        Start a new wandb run.

        Args:
            run_name: Name for the run (overrides experiment_name)

        Returns:
            Active wandb run
        """
        name = run_name or self.experiment_name

        self.run = wandb.init(
            project=self.project_name,
            group=self.group,
            entity=self.entity,
            name=name,
            config=self.config,
            reinit=True,
        )

        return self.run

    def end_run(self) -> None:
        """End the current wandb run."""
        if self.run:
            wandb.finish()
            self.run = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to wandb.

        Args:
            params: Dictionary of parameters to log
        """
        if self.run:
            # wandb accepts nested dictionaries directly
            wandb.config.update(params, allow_val_change=True)

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """
        Log metrics to wandb.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for the metrics
        """
        if self.run:
            wandb.log(metrics, step=step)

    def log_artifact(
        self, local_path: str, artifact_path: Optional[str] = None
    ) -> None:
        """
        Log an artifact to wandb.

        Args:
            local_path: Path to the artifact
            artifact_path: Path within the artifact directory (ignored in wandb)
        """
        if self.run:
            # For wandb, we use wandb.save for files to be uploaded
            wandb.save(local_path)

    def log_model(
        self, model: torch.nn.Module, artifact_path: str = "model", **kwargs
    ) -> str:
        """
        Log a PyTorch model to wandb.

        Args:
            model: PyTorch model to log
            artifact_path: Path within artifacts (used as prefix)
            **kwargs: Additional keyword arguments for wandb

        Returns:
            Identifier for the logged model
        """
        if self.run:
            # Save model state dict
            model_path = f"{artifact_path}.pt"
            torch.save(model.state_dict(), model_path)

            # Log model as artifact
            artifact = wandb.Artifact(
                name=f"model-{self.run.id}",
                type="model",
                metadata=kwargs.get("metadata", {}),
            )
            artifact.add_file(model_path)
            self.run.log_artifact(artifact)

            # Clean up
            os.remove(model_path)

            return f"wandb-artifact://{artifact.name}"

        return None

    def log_figure(self, figure: plt.Figure, filename: str) -> None:
        """
        Log a matplotlib figure to wandb.

        Args:
            figure: Matplotlib figure to log
            filename: Filename for the figure
        """
        if self.run:
            # wandb can log the figure directly
            wandb.log({os.path.splitext(filename)[0]: figure})

    def log_confusion_matrix(
        self,
        y_true: List[int],
        y_pred: List[int],
        class_names: List[str],
        normalize: bool = True,
        title: str = "Confusion Matrix",
        figure_size: tuple = (10, 8),
    ) -> None:
        """
        Log a confusion matrix visualization to wandb.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            class_names: List of class names
            normalize: Whether to normalize confusion matrix values
            title: Title for the confusion matrix plot
            figure_size: Size of the figure
        """
        if self.run:
            # Create confusion matrix
            cm = confusion_matrix(y_true, y_pred)

            if normalize:
                cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

            # Create figure and plot
            fig = plt.figure(figsize=figure_size)
            sns.heatmap(
                cm,
                annot=True,
                fmt=".2f" if normalize else "d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
            )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(title)
            plt.tight_layout()

            # Log directly to wandb
            wandb.log({"confusion_matrix": fig})

            # Close the figure
            plt.close(fig)


class MultiTracker(ExperimentTracker):
    """
    Tracker that combines multiple tracking platforms.
    """

    def __init__(self, trackers: List[ExperimentTracker]):
        """
        Initialize with a list of trackers.

        Args:
            trackers: List of ExperimentTracker instances
        """
        super().__init__(experiment_name="multi_tracker")
        self.trackers = trackers

    def start_run(self, run_name: Optional[str] = None) -> List[Any]:
        """
        Start a run on all trackers.

        Args:
            run_name: Name for the run

        Returns:
            List of active runs
        """
        return [tracker.start_run(run_name) for tracker in self.trackers]

    def end_run(self) -> None:
        """End the current run on all trackers."""
        for tracker in self.trackers:
            tracker.end_run()

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to all trackers.

        Args:
            params: Dictionary of parameters to log
        """
        for tracker in self.trackers:
            try:
                tracker.log_params(params)
            except Exception as e:
                logger.warning(
                    f"Failed to log params with tracker {type(tracker).__name__}: {e}"
                )

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """
        Log metrics to all trackers.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for the metrics
        """
        for tracker in self.trackers:
            try:
                tracker.log_metrics(metrics, step)
            except Exception as e:
                logger.warning(
                    f"Failed to log metrics with tracker {type(tracker).__name__}: {e}"
                )

    def log_artifact(
        self, local_path: str, artifact_path: Optional[str] = None
    ) -> None:
        """
        Log an artifact to all trackers.

        Args:
            local_path: Path to the artifact
            artifact_path: Path within the artifact directory
        """
        for tracker in self.trackers:
            try:
                tracker.log_artifact(local_path, artifact_path)
            except Exception as e:
                logger.warning(
                    f"Failed to log artifact with tracker {type(tracker).__name__}: {e}"
                )

    def log_model(self, model: torch.nn.Module, **kwargs) -> Dict[str, Any]:
        """
        Log a model to all trackers.

        Args:
            model: PyTorch model to log
            **kwargs: Additional keyword arguments for trackers

        Returns:
            Dictionary mapping tracker names to model URIs
        """
        results = {}
        for tracker in self.trackers:
            try:
                tracker_name = type(tracker).__name__
                results[tracker_name] = tracker.log_model(model, **kwargs)
            except Exception as e:
                logger.warning(
                    f"Failed to log model with tracker {type(tracker).__name__}: {e}"
                )

        return results

    def log_figure(self, figure: plt.Figure, filename: str) -> None:
        """
        Log a figure to all trackers.

        Args:
            figure: Matplotlib figure to log
            filename: Filename for the figure
        """
        for tracker in self.trackers:
            try:
                tracker.log_figure(figure, filename)
            except Exception as e:
                logger.warning(
                    f"Failed to log figure with tracker {type(tracker).__name__}: {e}"
                )

    def log_confusion_matrix(
        self, y_true: List[int], y_pred: List[int], class_names: List[str], **kwargs
    ) -> None:
        """
        Log a confusion matrix to all trackers.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            class_names: List of class names
            **kwargs: Additional keyword arguments
        """
        for tracker in self.trackers:
            try:
                tracker.log_confusion_matrix(y_true, y_pred, class_names, **kwargs)
            except Exception as e:
                logger.warning(
                    f"Failed to log confusion matrix with tracker {type(tracker).__name__}: {e}"
                )


def create_tracker(
    use_mlflow: bool = True,
    use_wandb: bool = True,
    experiment_name: str = "clip_har",
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_artifacts_dir: str = "./mlruns",
    wandb_project: str = "clip_har",
    wandb_group: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> ExperimentTracker:
    """
    Create a tracker based on specified options.

    Args:
        use_mlflow: Whether to use MLflow tracking
        use_wandb: Whether to use wandb tracking
        experiment_name: Name of the experiment
        mlflow_tracking_uri: MLflow tracking server URI
        mlflow_artifacts_dir: Local directory to store MLflow artifacts
        wandb_project: Name of the wandb project
        wandb_group: Group name for wandb
        wandb_entity: Team name or username for wandb
        config: Configuration dictionary for the run

    Returns:
        An ExperimentTracker instance
    """
    trackers = []

    if use_mlflow:
        try:
            trackers.append(
                MLflowTracker(
                    experiment_name=experiment_name,
                    tracking_uri=mlflow_tracking_uri,
                    artifacts_dir=mlflow_artifacts_dir,
                )
            )
        except Exception as e:
            logger.warning(f"Failed to create MLflowTracker: {e}")

    if use_wandb:
        try:
            if not WANDB_AVAILABLE:
                logger.warning("wandb is not installed, skipping wandb tracking")
            else:
                trackers.append(
                    WandbTracker(
                        experiment_name=experiment_name,
                        project_name=wandb_project,
                        group=wandb_group,
                        entity=wandb_entity,
                        config=config,
                    )
                )
        except Exception as e:
            logger.warning(f"Failed to create WandbTracker: {e}")

    if len(trackers) == 0:
        logger.warning("No trackers were created, falling back to MLflowTracker")
        trackers.append(
            MLflowTracker(
                experiment_name=experiment_name,
                tracking_uri=mlflow_tracking_uri,
                artifacts_dir=mlflow_artifacts_dir,
            )
        )

    if len(trackers) == 1:
        return trackers[0]
    else:
        return MultiTracker(trackers)


# For backwards compatibility
def setup_mlflow(
    experiment_name: str = "clip_har",
    tracking_uri: Optional[str] = None,
    artifacts_dir: str = "./mlruns",
) -> str:
    """
    Set up MLflow experiment tracking.

    Args:
        experiment_name: Name of the MLflow experiment
        tracking_uri: MLflow tracking server URI (None for local)
        artifacts_dir: Local directory to store artifacts

    Returns:
        The experiment ID
    """
    tracker = MLflowTracker(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        artifacts_dir=artifacts_dir,
    )
    return tracker.experiment_id


# For backwards compatibility - redirect to appropriate tracker methods
log_model_params = lambda params: logger.warning(
    "Use tracker.log_params() instead of log_model_params()"
)
log_metrics = lambda metrics, step=None: logger.warning(
    "Use tracker.log_metrics() instead of log_metrics()"
)
log_confusion_matrix = lambda y_true, y_pred, class_names, **kwargs: logger.warning(
    "Use tracker.log_confusion_matrix() instead of log_confusion_matrix()"
)
log_model = lambda model, **kwargs: logger.warning(
    "Use tracker.log_model() instead of log_model()"
)
log_artifact = lambda local_path, artifact_path=None: logger.warning(
    "Use tracker.log_artifact() instead of log_artifact()"
)


def log_model_predictions(
    image_paths: List[str],
    predictions: List[str],
    confidence_scores: List[float],
    sample_count: int = 10,
) -> None:
    """
    Log sample model predictions with images to MLflow.

    Args:
        image_paths: List of paths to the images
        predictions: List of predicted class names
        confidence_scores: List of confidence scores for predictions
        sample_count: Number of samples to log
    """
    if sample_count > len(image_paths):
        sample_count = len(image_paths)

    # Create a directory for the samples
    samples_dir = "prediction_samples"
    os.makedirs(samples_dir, exist_ok=True)

    # Create a metadata file
    metadata = []

    # Log sample images with predictions
    for i in range(sample_count):
        # Copy the image to the samples directory
        sample_name = f"sample_{i}.jpg"
        sample_path = os.path.join(samples_dir, sample_name)

        try:
            # Copy image file
            import shutil

            shutil.copy(image_paths[i], sample_path)

            # Record metadata
            metadata.append(
                {
                    "image": sample_name,
                    "prediction": predictions[i],
                    "confidence": float(confidence_scores[i]),
                }
            )
        except Exception as e:
            print(f"Error logging sample {i}: {e}")

    # Save metadata
    with open(os.path.join(samples_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Log the directory as an artifact
    mlflow.log_artifacts(samples_dir)

    # Clean up
    import shutil

    shutil.rmtree(samples_dir)


def load_model_from_mlflow(model_uri: str) -> torch.nn.Module:
    """
    Load a PyTorch model from MLflow.

    Args:
        model_uri: URI of the model to load

    Returns:
        Loaded PyTorch model
    """
    model = mlflow.pytorch.load_model(model_uri)
    return model
