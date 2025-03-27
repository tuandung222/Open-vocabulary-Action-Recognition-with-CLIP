import unittest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

class TestExperimentTracking(unittest.TestCase):
    """Unit tests for the experiment tracking functionality."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory for tracking artifacts
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock MLflow
        self.patcher_mlflow = patch('CLIP_HAR_PROJECT.mlops.tracking.mlflow')
        self.mock_mlflow = self.patcher_mlflow.start()
        
        # Mock wandb
        self.patcher_wandb = patch('CLIP_HAR_PROJECT.mlops.tracking.wandb')
        self.mock_wandb = self.patcher_wandb.start()
        
        # Set up dummy experiment info
        self.experiment_name = "test_experiment"
        self.config = {
            "model": {
                "name": "clip-vit-base",
                "prompt_template": "a photo of {}"
            },
            "training": {
                "epochs": 5,
                "batch_size": 32,
                "learning_rate": 1e-4
            }
        }
        self.metrics = {
            "accuracy": 0.92,
            "precision": 0.91,
            "recall": 0.90,
            "f1": 0.905
        }
        self.artifact_paths = {
            "model": os.path.join(self.temp_dir, "model.pt"),
            "confusion_matrix": os.path.join(self.temp_dir, "confusion_matrix.png")
        }
        
        # Create dummy artifact files
        with open(self.artifact_paths["model"], "w") as f:
            f.write("dummy model file")
        with open(self.artifact_paths["confusion_matrix"], "w") as f:
            f.write("dummy image file")
    
    def tearDown(self):
        """Clean up after each test."""
        self.patcher_mlflow.stop()
        self.patcher_wandb.stop()
        
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_mlflow_tracker(self):
        """Test MLflow tracker functionality."""
        from CLIP_HAR_PROJECT.mlops.tracking import MLflowTracker
        
        # Create MLflow tracker
        tracker = MLflowTracker(
            experiment_name=self.experiment_name,
            tracking_uri=None,
            artifacts_dir=self.temp_dir
        )
        
        # Start run
        tracker.start_run()
        
        # Log parameters, metrics, and artifacts
        tracker.log_params(self.config)
        tracker.log_metrics(self.metrics)
        tracker.log_artifacts(self.artifact_paths)
        
        # End run
        tracker.end_run()
        
        # Verify MLflow was used correctly
        self.mock_mlflow.set_experiment.assert_called_once_with(self.experiment_name)
        self.mock_mlflow.start_run.assert_called_once()
        self.mock_mlflow.log_params.assert_called()
        self.mock_mlflow.log_metrics.assert_called()
        self.mock_mlflow.log_artifact.assert_called()
        self.mock_mlflow.end_run.assert_called_once()
    
    def test_wandb_tracker(self):
        """Test Weights & Biases tracker functionality."""
        from CLIP_HAR_PROJECT.mlops.tracking import WandbTracker
        
        # Create wandb tracker
        tracker = WandbTracker(
            experiment_name=self.experiment_name,
            project_name="har-classification",
            group_name="clip-experiments",
            entity=None
        )
        
        # Start run
        tracker.start_run(config=self.config)
        
        # Log metrics and artifacts
        tracker.log_metrics(self.metrics)
        tracker.log_artifacts(self.artifact_paths)
        
        # End run
        tracker.end_run()
        
        # Verify wandb was used correctly
        self.mock_wandb.init.assert_called_once()
        self.mock_wandb.log.assert_called()
        self.mock_wandb.save.assert_called()
        self.mock_wandb.finish.assert_called_once()
    
    def test_multi_tracker(self):
        """Test the MultiTracker that combines different trackers."""
        from CLIP_HAR_PROJECT.mlops.tracking import MLflowTracker, WandbTracker, MultiTracker
        
        # Create individual trackers
        mlflow_tracker = MLflowTracker(
            experiment_name=self.experiment_name,
            tracking_uri=None,
            artifacts_dir=self.temp_dir
        )
        
        wandb_tracker = WandbTracker(
            experiment_name=self.experiment_name,
            project_name="har-classification",
            group_name="clip-experiments"
        )
        
        # Mock the individual trackers
        mlflow_tracker.start_run = MagicMock()
        mlflow_tracker.log_params = MagicMock()
        mlflow_tracker.log_metrics = MagicMock()
        mlflow_tracker.log_artifacts = MagicMock()
        mlflow_tracker.end_run = MagicMock()
        
        wandb_tracker.start_run = MagicMock()
        wandb_tracker.log_params = MagicMock()
        wandb_tracker.log_metrics = MagicMock()
        wandb_tracker.log_artifacts = MagicMock()
        wandb_tracker.end_run = MagicMock()
        
        # Create multi-tracker
        multi_tracker = MultiTracker([mlflow_tracker, wandb_tracker])
        
        # Test the multi-tracker
        multi_tracker.start_run()
        multi_tracker.log_params(self.config)
        multi_tracker.log_metrics(self.metrics)
        multi_tracker.log_artifacts(self.artifact_paths)
        multi_tracker.end_run()
        
        # Verify both trackers were used
        mlflow_tracker.start_run.assert_called_once()
        mlflow_tracker.log_params.assert_called_once_with(self.config)
        mlflow_tracker.log_metrics.assert_called_once_with(self.metrics)
        mlflow_tracker.log_artifacts.assert_called_once_with(self.artifact_paths)
        mlflow_tracker.end_run.assert_called_once()
        
        wandb_tracker.start_run.assert_called_once()
        wandb_tracker.log_params.assert_called_once_with(self.config)
        wandb_tracker.log_metrics.assert_called_once_with(self.metrics)
        wandb_tracker.log_artifacts.assert_called_once_with(self.artifact_paths)
        wandb_tracker.end_run.assert_called_once()
    
    def test_create_tracker_factory(self):
        """Test the create_tracker factory function."""
        from CLIP_HAR_PROJECT.mlops.tracking import create_tracker
        
        # Create tracker with both systems
        tracker = create_tracker(
            use_mlflow=True,
            use_wandb=True,
            experiment_name=self.experiment_name,
            mlflow_tracking_uri=None,
            mlflow_artifacts_dir=self.temp_dir,
            wandb_project="har-classification",
            wandb_group="clip-experiments",
            config=self.config
        )
        
        # Verify correct tracker type is returned
        from CLIP_HAR_PROJECT.mlops.tracking import MultiTracker
        self.assertIsInstance(tracker, MultiTracker)
        self.assertEqual(len(tracker.trackers), 2)
        
        # Create tracker with only MLflow
        tracker = create_tracker(
            use_mlflow=True,
            use_wandb=False,
            experiment_name=self.experiment_name,
            mlflow_tracking_uri=None,
            mlflow_artifacts_dir=self.temp_dir,
            config=self.config
        )
        
        # Verify correct tracker type is returned
        from CLIP_HAR_PROJECT.mlops.tracking import MLflowTracker
        self.assertIsInstance(tracker, MLflowTracker)
        
        # Create tracker with only wandb
        tracker = create_tracker(
            use_mlflow=False,
            use_wandb=True,
            experiment_name=self.experiment_name,
            wandb_project="har-classification",
            wandb_group="clip-experiments",
            config=self.config
        )
        
        # Verify correct tracker type is returned
        from CLIP_HAR_PROJECT.mlops.tracking import WandbTracker
        self.assertIsInstance(tracker, WandbTracker)


if __name__ == '__main__':
    unittest.main() 