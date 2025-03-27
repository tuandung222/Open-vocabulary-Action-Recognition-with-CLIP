import unittest
import torch
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

class TestModelExport(unittest.TestCase):
    """Unit tests for model export functionality."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory for outputs
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock the CLIP model
        self.patcher_model = patch('CLIP_HAR_PROJECT.models.clip_model.CLIPModel')
        self.mock_model_class = self.patcher_model.start()
        
        # Create a mock model instance
        self.mock_model = MagicMock()
        self.mock_model.state_dict.return_value = {"weights": torch.ones(1)}
        self.mock_model_class.return_value = self.mock_model
        
        # Mock the ONNX export function
        self.patcher_onnx = patch('CLIP_HAR_PROJECT.deployment.export.torch.onnx')
        self.mock_onnx = self.patcher_onnx.start()
        
        # Mock TensorRT import
        self.patcher_tensorrt = patch('CLIP_HAR_PROJECT.deployment.export.tensorrt')
        self.mock_tensorrt = self.patcher_tensorrt.start()
        
        # Mock torch.jit for TorchScript export
        self.patcher_jit = patch('CLIP_HAR_PROJECT.deployment.export.torch.jit')
        self.mock_jit = self.patcher_jit.start()
        
        # Mock class names for testing
        self.class_names = [f"class_{i}" for i in range(15)]
    
    def tearDown(self):
        """Clean up after each test."""
        # Stop all patches
        self.patcher_model.stop()
        self.patcher_onnx.stop()
        self.patcher_tensorrt.stop()
        self.patcher_jit.stop()
        
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_onnx_export(self):
        """Test exporting model to ONNX format."""
        from CLIP_HAR_PROJECT.deployment.export import export_to_onnx
        
        # Export the mock model to ONNX
        onnx_path = os.path.join(self.temp_dir, "model.onnx")
        
        result = export_to_onnx(
            model=self.mock_model,
            output_path=onnx_path,
            input_shape=(1, 3, 224, 224),
            dynamic_axes=True
        )
        
        # Check that the export was successful
        self.assertTrue(result)
        
        # Verify that torch.onnx.export was called with correct parameters
        self.mock_onnx.export.assert_called_once()
        
        # Verify dynamic axes were set correctly if specified
        _, call_kwargs = self.mock_onnx.export.call_args
        self.assertIn('dynamic_axes', call_kwargs)
    
    def test_torchscript_export(self):
        """Test exporting model to TorchScript format."""
        from CLIP_HAR_PROJECT.deployment.export import export_to_torchscript
        
        # Export the mock model to TorchScript
        script_path = os.path.join(self.temp_dir, "model.pt")
        
        result = export_to_torchscript(
            model=self.mock_model,
            output_path=script_path,
            input_shape=(1, 3, 224, 224)
        )
        
        # Check that the export was successful
        self.assertTrue(result)
        
        # Verify that torch.jit.trace was called with correct parameters
        self.mock_jit.trace.assert_called_once()
        
        # Verify save was called
        self.mock_jit.trace.return_value.save.assert_called_once_with(script_path)
    
    @patch('CLIP_HAR_PROJECT.deployment.export.os.path.exists')
    def test_export_model_meta(self, mock_exists):
        """Test exporting model metadata with the model."""
        from CLIP_HAR_PROJECT.deployment.export import export_model_metadata
        
        # Mock file existence check
        mock_exists.return_value = True
        
        # Export metadata
        meta_path = os.path.join(self.temp_dir, "metadata.json")
        
        result = export_model_metadata(
            model=self.mock_model,
            output_path=meta_path,
            class_names=self.class_names,
            model_type="clip",
            input_shape=(3, 224, 224)
        )
        
        # Check that the export was successful
        self.assertTrue(result)
    
    @patch('CLIP_HAR_PROJECT.deployment.export.export_to_onnx')
    @patch('CLIP_HAR_PROJECT.deployment.export.export_to_torchscript')
    @patch('CLIP_HAR_PROJECT.deployment.export.export_model_metadata')
    def test_export_model_all_formats(self, mock_metadata, mock_ts, mock_onnx):
        """Test exporting model to all supported formats."""
        from CLIP_HAR_PROJECT.deployment.export import export_model
        
        # Set up mocks to return success
        mock_onnx.return_value = True
        mock_ts.return_value = True
        mock_metadata.return_value = True
        
        # Test exporting to multiple formats
        result = export_model(
            model=self.mock_model,
            output_dir=self.temp_dir,
            formats=["onnx", "torchscript"],
            class_names=self.class_names,
            input_shape=(3, 224, 224)
        )
        
        # Check that all formats were exported
        self.assertEqual(len(result), 2)
        self.assertIn("onnx", result)
        self.assertIn("torchscript", result)
        
        # Verify individual export functions were called
        mock_onnx.assert_called_once()
        mock_ts.assert_called_once()
        mock_metadata.assert_called_once()


class TestInferenceAdapters(unittest.TestCase):
    """Unit tests for inference adapters for different model formats."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory for models
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock PyTorch model
        self.mock_pt_model = MagicMock()
        self.mock_pt_model.return_value = torch.ones(1, 15)
        
        # Create dummy input tensor
        self.dummy_input = torch.ones(1, 3, 224, 224)
        
        # Mock ONNX Runtime
        self.patcher_ort = patch('CLIP_HAR_PROJECT.mlops.inference_serving.ort')
        self.mock_ort = self.patcher_ort.start()
        self.mock_ort.InferenceSession.return_value.run.return_value = [
            np.ones((1, 15), dtype=np.float32)
        ]
        
        # Mock TensorRT engine
        self.patcher_trt = patch('CLIP_HAR_PROJECT.mlops.inference_serving.tensorrt')
        self.mock_trt = self.patcher_trt.start()
        
        # Mock TorchScript module
        self.patcher_jit = patch('CLIP_HAR_PROJECT.mlops.inference_serving.torch.jit')
        self.mock_jit = self.patcher_jit.start()
        self.mock_jit.load.return_value = self.mock_pt_model
        
    def tearDown(self):
        """Clean up after each test."""
        # Stop all patches
        self.patcher_ort.stop()
        self.patcher_trt.stop()
        self.patcher_jit.stop()
        
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_pytorch_adapter(self):
        """Test PyTorch model adapter."""
        from CLIP_HAR_PROJECT.mlops.inference_serving import PyTorchAdapter
        
        # Create adapter
        adapter = PyTorchAdapter(
            model=self.mock_pt_model,
            device="cpu"
        )
        
        # Test inference
        outputs = adapter.predict(self.dummy_input)
        
        # Verify model was called with correct input
        self.mock_pt_model.assert_called_once()
        self.assertIsNotNone(outputs)
    
    def test_onnx_adapter(self):
        """Test ONNX model adapter."""
        from CLIP_HAR_PROJECT.mlops.inference_serving import ONNXAdapter
        
        # Create model path
        model_path = os.path.join(self.temp_dir, "model.onnx")
        
        # Create adapter
        adapter = ONNXAdapter(
            model_path=model_path,
            input_name="input",
            output_name="output"
        )
        
        # Test inference
        outputs = adapter.predict(self.dummy_input)
        
        # Verify ONNX session was created and run
        self.mock_ort.InferenceSession.assert_called_once()
        self.mock_ort.InferenceSession.return_value.run.assert_called_once()
        self.assertIsNotNone(outputs)
    
    def test_torchscript_adapter(self):
        """Test TorchScript model adapter."""
        from CLIP_HAR_PROJECT.mlops.inference_serving import TorchScriptAdapter
        
        # Create model path
        model_path = os.path.join(self.temp_dir, "model.pt")
        
        # Create adapter
        adapter = TorchScriptAdapter(
            model_path=model_path,
            device="cpu"
        )
        
        # Test inference
        outputs = adapter.predict(self.dummy_input)
        
        # Verify TorchScript model was loaded and called
        self.mock_jit.load.assert_called_once()
        self.mock_pt_model.assert_called_once()
        self.assertIsNotNone(outputs)


if __name__ == '__main__':
    unittest.main() 