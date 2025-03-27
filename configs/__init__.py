from .default import (
    DataConfig,
    LoggingConfig,
    ModelConfig,
    ProjectConfig,
    TrainingConfig,
    get_auto_config,
    get_config,
    get_ddp_config,
    get_fsdp_config,
    get_single_gpu_config,
)

from .custom_config import get_quick_test_config
