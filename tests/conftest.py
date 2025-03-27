import os
import pytest
import torch
import tempfile
import shutil
from unittest.mock import MagicMock, patch

@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory that persists for the test session."""
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)

@pytest.fixture
def output_dir(temp_dir):
    """Create a temporary output directory."""
    output_path = os.path.join(temp_dir, "outputs")
    os.makedirs(output_path, exist_ok=True)
    return output_path

@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    dataset = MagicMock()
    dataset.__len__.return_value = 100
    
    # Mock getitem to return dummy data
    dataset.__getitem__.return_value = (
        torch.ones(3, 224, 224), 
        torch.tensor(0)
    )
    
    return dataset

@pytest.fixture
def mock_dataloader(mock_dataset):
    """Create a mock dataloader for testing."""
    with patch('torch.utils.data.DataLoader') as mock_loader:
        mock_loader.return_value = [
            (torch.ones(8, 3, 224, 224), torch.randint(0, 15, (8,)))
            for _ in range(5)
        ]
        yield mock_loader

@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    model.return_value = torch.ones(8, 15)
    model.extract_features.return_value = torch.ones(8, 512)
    return model

@pytest.fixture
def mock_tracker():
    """Create a mock experiment tracker for testing."""
    tracker = MagicMock()
    return tracker

@pytest.fixture
def basic_config():
    """Return a basic configuration dictionary for testing."""
    return {
        "model": {
            "name": "ViT-B/32",
            "num_classes": 15
        },
        "training": {
            "batch_size": 8,
            "epochs": 2,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "warmup_steps": 100
        },
        "data": {
            "dataset_name": "dummy",
            "image_size": 224,
            "train_ratio": 0.8,
            "val_ratio": 0.1
        },
        "augmentation": {
            "strength": "medium"
        }
    }

@pytest.fixture
def mock_clip():
    """Mock the CLIP model import."""
    with patch('CLIP_HAR_PROJECT.models.clip_model.clip') as mock_clip:
        # Set up mock returns
        mock_clip.load.return_value = (MagicMock(), MagicMock())
        mock_clip.tokenize.return_value = torch.ones((1, 77))
        yield mock_clip

@pytest.fixture
def mock_distributed_env():
    """Create a mock distributed environment."""
    return {
        "rank": 0,
        "world_size": 2,
        "local_rank": 0,
        "device": torch.device("cpu")
    }

@pytest.fixture
def disable_cuda_for_testing():
    """Disable CUDA for testing."""
    original_cuda = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    yield
    torch.cuda.is_available = original_cuda 