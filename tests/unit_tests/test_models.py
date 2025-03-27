import unittest
import torch
from unittest.mock import MagicMock, patch

class TestCLIPModel(unittest.TestCase):
    """Unit tests for the CLIP model implementation."""

    def setUp(self):
        """Set up test environment before each test."""
        # Mock the CLIP model import to avoid actual model loading
        self.clip_model_patcher = patch('CLIP_HAR_PROJECT.models.clip_model.clip')
        self.mock_clip = self.clip_model_patcher.start()
        
        # Set up mock returns
        self.mock_clip.load.return_value = (MagicMock(), MagicMock())
        self.mock_clip.tokenize.return_value = torch.ones((1, 77))
        
        # Import after patching
        from CLIP_HAR_PROJECT.models.clip_model import CLIPModel
        self.model_class = CLIPModel
    
    def tearDown(self):
        """Clean up after each test."""
        self.clip_model_patcher.stop()
    
    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        model = self.model_class(
            model_name="ViT-B/32",
            num_classes=15,
            device="cpu"
        )
        
        self.assertEqual(model.num_classes, 15)
        self.assertEqual(model.device, "cpu")
        self.mock_clip.load.assert_called_once_with("ViT-B/32", device="cpu")
    
    def test_forward_pass(self):
        """Test the forward pass of the model."""
        model = self.model_class(
            model_name="ViT-B/32",
            num_classes=15,
            device="cpu"
        )
        
        # Mock the CLIP model's forward and feature extraction
        model.clip_model.encode_image.return_value = torch.ones((1, 512))
        model.classifier.forward.return_value = torch.ones((1, 15))
        
        # Create dummy input
        input_tensor = torch.ones((1, 3, 224, 224))
        
        # Run forward pass
        output = model.forward(input_tensor)
        
        # Check output shape
        self.assertEqual(output.shape, (1, 15))
        
        # Verify the CLIP model was called correctly
        model.clip_model.encode_image.assert_called_once()
        model.classifier.forward.assert_called_once()
    
    def test_extract_features(self):
        """Test feature extraction from the model."""
        model = self.model_class(
            model_name="ViT-B/32", 
            num_classes=15,
            device="cpu"
        )
        
        # Mock the CLIP model's feature extraction
        model.clip_model.encode_image.return_value = torch.ones((1, 512))
        
        # Create dummy input
        input_tensor = torch.ones((1, 3, 224, 224))
        
        # Run feature extraction
        features = model.extract_features(input_tensor)
        
        # Check output shape
        self.assertEqual(features.shape, (1, 512))
        
        # Verify the CLIP model was called correctly
        model.clip_model.encode_image.assert_called_once()
    
    def test_zero_shot_classification(self):
        """Test zero-shot classification with text prompts."""
        model = self.model_class(
            model_name="ViT-B/32",
            num_classes=15, 
            device="cpu"
        )
        
        # Mock the CLIP model's encoding
        model.clip_model.encode_image.return_value = torch.ones((1, 512))
        model.clip_model.encode_text.return_value = torch.ones((15, 512))
        
        # Create dummy input and class names
        input_tensor = torch.ones((1, 3, 224, 224))
        class_names = ["class_{}".format(i) for i in range(15)]
        
        # Run zero-shot classification
        logits_per_image, probs = model.zero_shot_classification(
            input_tensor, 
            class_names,
            template="a photo of a person {}"
        )
        
        # Check output shapes
        self.assertEqual(logits_per_image.shape, (1, 15))
        self.assertEqual(probs.shape, (1, 15))
        
        # Verify the CLIP model and tokenize were called correctly
        model.clip_model.encode_image.assert_called_once()
        model.clip_model.encode_text.assert_called_once()
        self.mock_clip.tokenize.assert_called()


if __name__ == '__main__':
    unittest.main() 