import unittest
import os
import tempfile
import shutil
import torch
from unittest.mock import patch, MagicMock

class TestTrainingPipeline(unittest.TestCase):
    """Integration tests for the training pipeline."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory for outputs
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "outputs")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up mock dataset
        self.mock_dataset = MagicMock()
        self.mock_dataset.__len__.return_value = 100
        
        # Mock DataLoader to return dummy batches
        self.patcher_dataloader = patch('torch.utils.data.DataLoader')
        self.mock_dataloader = self.patcher_dataloader.start()
        self.mock_dataloader.return_value = [
            (torch.ones(8, 3, 224, 224), torch.randint(0, 15, (8,)))
            for _ in range(5)
        ]
        
        # Patch the model creation
        self.patcher_model = patch('CLIP_HAR_PROJECT.models.model_factory.create_model')
        self.mock_create_model = self.patcher_model.start()
        self.mock_model = MagicMock()
        self.mock_model.return_value = torch.ones(8, 15)
        self.mock_create_model.return_value = self.mock_model
        
        # Patch the experiment tracker
        self.patcher_tracker = patch('CLIP_HAR_PROJECT.mlops.tracking.create_experiment_tracker')
        self.mock_create_tracker = self.patcher_tracker.start()
        self.mock_tracker = MagicMock()
        self.mock_create_tracker.return_value = self.mock_tracker
    
    def tearDown(self):
        """Clean up after each test."""
        self.patcher_dataloader.stop()
        self.patcher_model.stop()
        self.patcher_tracker.stop()
        
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    @patch('CLIP_HAR_PROJECT.data.dataset.load_dataset')
    def test_pipeline_execution(self, mock_load_dataset):
        """Test the entire training pipeline execution."""
        from CLIP_HAR_PROJECT.pipeline.training_pipeline import TrainingPipeline
        
        # Setup mock dataset
        mock_load_dataset.return_value = (self.mock_dataset, self.mock_dataset, self.mock_dataset)
        
        # Initialize pipeline
        pipeline = TrainingPipeline(
            config={
                "model": {
                    "name": "ViT-B/32",
                    "num_classes": 15
                },
                "training": {
                    "batch_size": 8,
                    "epochs": 2,
                    "learning_rate": 1e-4
                },
                "data": {
                    "dataset_name": "dummy",
                    "image_size": 224
                }
            },
            output_dir=self.output_dir,
            distributed_mode="none",
            device="cpu",
            use_mlflow=False,
            use_wandb=False
        )
        
        # Run pipeline
        results = pipeline.run()
        
        # Check that the pipeline ran successfully
        self.assertIsNotNone(results)
        self.assertIn("accuracy", results)
        self.assertIn("loss", results)
        
        # Verify model was used
        self.mock_model.train.assert_called()
        self.mock_model.eval.assert_called()
        
        # Verify model was saved
        self.mock_model.save.assert_called()
        
        # Verify tracker was used
        self.mock_tracker.log_params.assert_called()
        self.mock_tracker.log_metrics.assert_called()

    @patch('CLIP_HAR_PROJECT.data.dataset.load_dataset')
    def test_pipeline_with_tracking(self, mock_load_dataset):
        """Test the training pipeline with experiment tracking."""
        from CLIP_HAR_PROJECT.pipeline.training_pipeline import TrainingPipeline
        
        # Setup mock dataset
        mock_load_dataset.return_value = (self.mock_dataset, self.mock_dataset, self.mock_dataset)
        
        # Initialize pipeline with tracking
        pipeline = TrainingPipeline(
            config={
                "model": {
                    "name": "ViT-B/32",
                    "num_classes": 15
                },
                "training": {
                    "batch_size": 8,
                    "epochs": 2,
                    "learning_rate": 1e-4
                },
                "data": {
                    "dataset_name": "dummy",
                    "image_size": 224
                }
            },
            output_dir=self.output_dir,
            distributed_mode="none",
            device="cpu",
            use_mlflow=True,
            use_wandb=True,
            experiment_name="test_experiment"
        )
        
        # Run pipeline
        pipeline.run()
        
        # Verify experiment was created
        self.mock_create_tracker.assert_called_with(
            use_mlflow=True,
            use_wandb=True,
            experiment_name="test_experiment",
            config=pipeline.config
        )
        
        # Verify tracker methods were called
        self.mock_tracker.log_params.assert_called()
        self.mock_tracker.log_metrics.assert_called()
        self.mock_tracker.log_artifacts.assert_called()
        self.mock_tracker.finalize.assert_called()

    @patch('CLIP_HAR_PROJECT.training.distributed.setup_distributed')
    @patch('CLIP_HAR_PROJECT.data.dataset.load_dataset')
    def test_pipeline_distributed(self, mock_load_dataset, mock_setup_distributed):
        """Test the training pipeline in distributed mode."""
        from CLIP_HAR_PROJECT.pipeline.training_pipeline import TrainingPipeline
        
        # Setup mock dataset
        mock_load_dataset.return_value = (self.mock_dataset, self.mock_dataset, self.mock_dataset)
        
        # Mock distributed setup
        mock_setup_distributed.return_value = {
            "rank": 0,
            "world_size": 2,
            "local_rank": 0,
            "device": torch.device("cpu")
        }
        
        # Initialize pipeline with distributed mode
        pipeline = TrainingPipeline(
            config={
                "model": {
                    "name": "ViT-B/32",
                    "num_classes": 15
                },
                "training": {
                    "batch_size": 8,
                    "epochs": 2,
                    "learning_rate": 1e-4
                },
                "data": {
                    "dataset_name": "dummy",
                    "image_size": 224
                }
            },
            output_dir=self.output_dir,
            distributed_mode="ddp",
            device="cpu",
            use_mlflow=False,
            use_wandb=False
        )
        
        # Run pipeline
        pipeline.run()
        
        # Verify distributed setup was called
        mock_setup_distributed.assert_called_with("ddp")
        
        # Verify model was wrapped in DDP
        self.mock_create_model.assert_called()


if __name__ == '__main__':
    unittest.main() 