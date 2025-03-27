from CLIP_HAR_PROJECT.configs.default import ProjectConfig, ModelConfig, DataConfig, TrainingConfig, LoggingConfig
from dataclasses import field

def get_quick_test_config() -> ProjectConfig:
    """
    Get a configuration for quick testing with only 5 epochs.
    
    Returns:
        ProjectConfig: Configuration optimized for quick testing
    """
    config = ProjectConfig()
    
    # Reduce training epochs to 5
    config.training.max_epochs = 5
    
    # Use smaller batch size for faster iteration
    config.training.batch_size = 64
    
    # Adjust other training parameters
    config.training.output_dir = "outputs/quick_test"
    
    # Customize logging
    config.logging.project_name = "clip_har"
    config.logging.group_name = "quick_tests"
    config.logging.experiment_name = "quick_test_5epochs"
    
    return config 