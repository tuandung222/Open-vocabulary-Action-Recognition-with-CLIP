import unittest
import os
import shutil
import tempfile
import torch
from unittest.mock import patch, MagicMock

class TestEndToEndWorkflow(unittest.TestCase):
    """
    Integration test for the complete workflow from training to deployment.
    
    This test verifies the entire pipeline:
    1. Training a model
    2. Evaluating the model
    3. Exporting to deployment formats
    4. Setting up inference serving
    5. Optionally pushing to Hugging Face Hub
    """
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory for all outputs
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "outputs")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Mock datasets
        self.mock_train_dataset = MagicMock()
        self.mock_train_dataset.__len__.return_value = 100
        self.mock_val_dataset = MagicMock()
        self.mock_val_dataset.__len__.return_value = 20
        self.mock_test_dataset = MagicMock()
        self.mock_test_dataset.__len__.return_value = 20
        
        # Mock dataloaders to return dummy batches
        self.patcher_dataloader = patch('torch.utils.data.DataLoader')
        self.mock_dataloader = self.patcher_dataloader.start()
        self.mock_dataloader.return_value = [
            (torch.ones(8, 3, 224, 224), torch.randint(0, 15, (8,)))
            for _ in range(3)
        ]
        
        # Patch dataset loading to return our mock datasets
        self.patcher_dataset = patch('CLIP_HAR_PROJECT.data.dataset.load_har_dataset')
        self.mock_load_dataset = self.patcher_dataset.start()
        self.mock_load_dataset.return_value = (
            self.mock_train_dataset,
            self.mock_val_dataset,
            self.mock_test_dataset
        )
        
        # Mock config to use for tests
        self.test_config = {
            "model": {
                "model_name": "openai/clip-vit-base-patch16",
                "prompt_template": "a photo of a person {}",
                "unfreeze_visual_encoder": False,
                "unfreeze_text_encoder": False
            },
            "training": {
                "max_epochs": 1,
                "batch_size": 8,
                "eval_batch_size": 8,
                "lr": 1e-4,
                "output_dir": self.output_dir,
                "mixed_precision": True,
                "distributed_mode": "none"
            },
            "data": {
                "dataset_name": "dummy",
                "val_ratio": 0.2,
                "test_ratio": 0.2,
                "seed": 42,
                "image_size": 224
            },
            "logging": {
                "use_wandb": False,
                "use_mlflow": False
            }
        }
        
        # Mock model creation
        self.patcher_model = patch('CLIP_HAR_PROJECT.models.model_factory.create_model')
        self.mock_create_model = self.patcher_model.start()
        self.mock_model = MagicMock()
        self.mock_model.forward.return_value = torch.ones(8, 15)
        self.mock_model.extract_features.return_value = torch.ones(8, 512)
        self.mock_create_model.return_value = self.mock_model
        
        # Mock ONNX export
        self.patcher_onnx = patch('CLIP_HAR_PROJECT.deployment.export.export_to_onnx')
        self.mock_onnx_export = self.patcher_onnx.start()
        self.mock_onnx_export.return_value = True
        
        # Mock TorchScript export
        self.patcher_script = patch('CLIP_HAR_PROJECT.deployment.export.export_to_torchscript')
        self.mock_script_export = self.patcher_script.start()
        self.mock_script_export.return_value = True
        
        # Mock inference server
        self.patcher_server = patch('CLIP_HAR_PROJECT.mlops.inference_serving.InferenceServer')
        self.mock_server_class = self.patcher_server.start()
        self.mock_server = MagicMock()
        self.mock_server_class.return_value = self.mock_server
        
        # Mock HuggingFace Hub push
        self.patcher_hub = patch('CLIP_HAR_PROJECT.mlops.huggingface_hub_utils.push_model_to_hub')
        self.mock_hub_push = self.patcher_hub.start()
        self.mock_hub_push.return_value = "https://huggingface.co/test/model"
    
    def tearDown(self):
        """Clean up after each test."""
        # Stop all patches
        self.patcher_dataloader.stop()
        self.patcher_dataset.stop()
        self.patcher_model.stop()
        self.patcher_onnx.stop()
        self.patcher_script.stop()
        self.patcher_server.stop()
        self.patcher_hub.stop()
        
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    @patch('CLIP_HAR_PROJECT.configs.ProjectConfig')
    def test_training_pipeline(self, mock_config_class):
        """Test the training pipeline."""
        from CLIP_HAR_PROJECT.pipeline.training_pipeline import TrainingPipeline
        
        # Mock the config class
        mock_config = MagicMock()
        for section, values in self.test_config.items():
            setattr(mock_config, section, MagicMock(**values))
        mock_config_class.return_value = mock_config
        
        # Create the pipeline
        pipeline = TrainingPipeline(
            config=mock_config,
            distributed_mode="none",
            use_mlflow=False,
            use_wandb=False
        )
        
        # Run the training
        results = pipeline.run()
        
        # Verify that training completed successfully
        self.assertIsNotNone(results)
        self.assertIn("training_completed", results)
        self.assertTrue(results["training_completed"])
    
    @patch('CLIP_HAR_PROJECT.configs.ProjectConfig')
    def test_export_after_training(self, mock_config_class):
        """Test exporting a model after training."""
        from CLIP_HAR_PROJECT.pipeline.training_pipeline import TrainingPipeline
        from CLIP_HAR_PROJECT.deployment.export import export_model
        
        # Mock the config class
        mock_config = MagicMock()
        for section, values in self.test_config.items():
            setattr(mock_config, section, MagicMock(**values))
        mock_config_class.return_value = mock_config
        
        # Create the pipeline
        pipeline = TrainingPipeline(
            config=mock_config,
            distributed_mode="none",
            use_mlflow=False,
            use_wandb=False
        )
        
        # Run the training
        pipeline.run()
        
        # Export the model
        export_paths = export_model(
            model=self.mock_model,
            output_dir=os.path.join(self.output_dir, "exports"),
            formats=["onnx", "torchscript"],
            class_names=["class_" + str(i) for i in range(15)],
            input_shape=(3, 224, 224)
        )
        
        # Verify that export calls were made
        self.mock_onnx_export.assert_called_once()
        self.mock_script_export.assert_called_once()
        
        # Check that export paths were returned
        self.assertIn("onnx", export_paths)
        self.assertIn("torchscript", export_paths)
    
    @patch('CLIP_HAR_PROJECT.configs.ProjectConfig')
    def test_inference_serving(self, mock_config_class):
        """Test setting up inference serving after training and export."""
        from CLIP_HAR_PROJECT.pipeline.training_pipeline import TrainingPipeline
        from CLIP_HAR_PROJECT.deployment.export import export_model
        from CLIP_HAR_PROJECT.mlops.inference_serving import setup_inference_server
        
        # Mock the config class
        mock_config = MagicMock()
        for section, values in self.test_config.items():
            setattr(mock_config, section, MagicMock(**values))
        mock_config_class.return_value = mock_config
        
        # Create the pipeline
        pipeline = TrainingPipeline(
            config=mock_config,
            distributed_mode="none",
            use_mlflow=False,
            use_wandb=False
        )
        
        # Run the training
        pipeline.run()
        
        # Export the model
        export_paths = export_model(
            model=self.mock_model,
            output_dir=os.path.join(self.output_dir, "exports"),
            formats=["onnx"],
            class_names=["class_" + str(i) for i in range(15)],
            input_shape=(3, 224, 224)
        )
        
        # Setup inference server with the ONNX model
        server = setup_inference_server(
            model_path=export_paths["onnx"],
            model_type="onnx",
            class_names=["class_" + str(i) for i in range(15)],
            device="cpu",
            host="localhost",
            port=8000
        )
        
        # Verify that the server was set up correctly
        self.mock_server_class.assert_called_once()
        
        # Start the server in the background
        server.start_background()
        
        # Verify start method was called
        self.mock_server.start_background.assert_called_once()
    
    @patch('CLIP_HAR_PROJECT.configs.ProjectConfig')
    def test_complete_workflow_with_hub_push(self, mock_config_class):
        """Test the complete workflow including pushing to Hugging Face Hub."""
        from CLIP_HAR_PROJECT.pipeline.training_pipeline import TrainingPipeline
        from CLIP_HAR_PROJECT.deployment.export import export_model
        from CLIP_HAR_PROJECT.mlops.huggingface_hub_utils import push_model_to_hub
        
        # Mock the config class
        mock_config = MagicMock()
        for section, values in self.test_config.items():
            setattr(mock_config, section, MagicMock(**values))
        mock_config_class.return_value = mock_config
        
        # Create the pipeline
        pipeline = TrainingPipeline(
            config=mock_config,
            distributed_mode="none",
            use_mlflow=False,
            use_wandb=False
        )
        
        # Run the training
        results = pipeline.run()
        
        # Export the model
        export_paths = export_model(
            model=self.mock_model,
            output_dir=os.path.join(self.output_dir, "exports"),
            formats=["onnx", "torchscript"],
            class_names=["class_" + str(i) for i in range(15)],
            input_shape=(3, 224, 224)
        )
        
        # Push to Hugging Face Hub
        hub_url = push_model_to_hub(
            model=self.mock_model,
            repo_id="test/clip-har-model",
            commit_message="Test model upload",
            model_name="clip-har-test",
            metadata={
                "accuracy": 0.95,
                "f1_score": 0.94,
                "framework": "pytorch",
                "is_test": True
            }
        )
        
        # Verify hub push was called
        self.mock_hub_push.assert_called_once()
        
        # Check the hub URL
        self.assertEqual(hub_url, "https://huggingface.co/test/model")


if __name__ == '__main__':
    unittest.main() 