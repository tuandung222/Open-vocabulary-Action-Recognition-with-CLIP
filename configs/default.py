import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ModelConfig:
    """Configuration for the model."""

    model_name: str = "openai/clip-vit-base-patch16"
    prompt_template: str = "a photo of person/people who is/are {label}"
    unfreeze_visual_encoder: bool = True
    unfreeze_text_encoder: bool = False


@dataclass
class DataConfig:
    """Configuration for the data."""

    dataset_name: str = "Bingsu/Human_Action_Recognition"
    val_ratio: float = 0.15
    test_ratio: float = 0.25
    seed: int = 42
    image_size: int = 224


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Basic training parameters
    max_epochs: int = 15
    lr: float = 3e-6
    betas: Tuple[float, float] = (0.9, 0.995)
    weight_decay: float = 0.01
    num_warmup_steps: int = 3
    train_log_interval: int = 10
    val_log_interval: int = 10
    max_patience: int = 5

    # Output directories
    output_dir: str = "checkpoints"

    # Batch sizes
    batch_size: int = 256
    eval_batch_size: int = 128
    num_workers: int = 2

    # Mixed precision training
    mixed_precision: bool = True

    # Distributed training
    distributed_mode: str = "none"  # "none", "ddp", or "fsdp"

    def to_dict(self):
        """Convert to dictionary."""
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    use_wandb: bool = True
    project_name: str = "har_classification"
    group_name: str = "clip_training"
    experiment_name: Optional[str] = None  # Will be auto-generated if None

    # Visualization settings
    num_samples_to_visualize: int = 15
    confusion_matrix: bool = True
    class_distribution: bool = True


@dataclass
class ProjectConfig:
    """Main configuration for the project."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# Predefined configurations
def get_single_gpu_config() -> ProjectConfig:
    """Get configuration for single GPU training."""
    config = ProjectConfig()
    config.training.distributed_mode = "none"
    config.training.batch_size = 128
    return config


def get_ddp_config() -> ProjectConfig:
    """Get configuration for DDP training."""
    config = ProjectConfig()
    config.training.distributed_mode = "ddp"
    config.training.batch_size = 64  # Per GPU
    return config


def get_fsdp_config() -> ProjectConfig:
    """Get configuration for FSDP training."""
    config = ProjectConfig()
    config.training.distributed_mode = "fsdp"
    config.training.batch_size = 32  # Per GPU, smaller due to memory overhead
    return config


def get_config(distributed_mode: str = "none") -> ProjectConfig:
    """
    Get configuration based on distributed mode.

    Args:
        distributed_mode: One of "none", "ddp", or "fsdp"

    Returns:
        The configuration for the specified mode
    """
    if distributed_mode.lower() == "ddp":
        return get_ddp_config()
    elif distributed_mode.lower() == "fsdp":
        return get_fsdp_config()
    else:
        return get_single_gpu_config()


# Environment-based automatic configuration
def get_auto_config() -> ProjectConfig:
    """
    Automatically determine the best configuration based on the environment.

    Returns:
        The best configuration for the current environment
    """
    # Check for distributed environment variables
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        # Multi-GPU environment detected
        return get_ddp_config()
    else:
        # Single GPU environment
        return get_single_gpu_config()
