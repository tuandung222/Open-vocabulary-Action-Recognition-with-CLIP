import unittest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

class TestDataPipeline(unittest.TestCase):
    """Unit tests for the data processing pipeline."""

    def setUp(self):
        """Set up test environment before each test."""
        # Mock the dataset loading to avoid actual network calls
        self.patcher_dataset = patch('CLIP_HAR_PROJECT.data.dataset.datasets')
        self.mock_datasets = self.patcher_dataset.start()
        
        # Create mock dataset
        self.mock_dataset = MagicMock()
        self.mock_dataset.features = {'image': MagicMock(), 'label': MagicMock()}
        self.mock_dataset.__len__.return_value = 100
        
        # Mock image and label data
        mock_images = [np.random.rand(224, 224, 3) for _ in range(100)]
        mock_labels = [np.random.randint(0, 15) for _ in range(100)]
        
        self.mock_dataset.__getitem__.side_effect = lambda idx: {
            'image': mock_images[idx],
            'label': mock_labels[idx]
        }
        
        # Set up mock return for datasets.load_dataset
        self.mock_datasets.load_dataset.return_value = self.mock_dataset
        
        # Mock transformers for tokenization and image processing
        self.patcher_tokenizer = patch('CLIP_HAR_PROJECT.data.preprocessing.CLIPTokenizer')
        self.mock_tokenizer = self.patcher_tokenizer.start()
        self.mock_tokenizer.from_pretrained.return_value.encode.return_value = torch.ones((1, 77))
        
        self.patcher_processor = patch('CLIP_HAR_PROJECT.data.preprocessing.CLIPProcessor')
        self.mock_processor = self.patcher_processor.start()
        
    def tearDown(self):
        """Clean up after each test."""
        self.patcher_dataset.stop()
        self.patcher_tokenizer.stop()
        self.patcher_processor.stop()
    
    def test_dataset_loading(self):
        """Test that the dataset is loaded correctly."""
        from CLIP_HAR_PROJECT.data.dataset import load_har_dataset
        
        # Call the function
        train_ds, val_ds, test_ds = load_har_dataset(
            dataset_name="dummy",
            val_ratio=0.1,
            test_ratio=0.1
        )
        
        # Check that the function returns correctly split datasets
        self.assertIsNotNone(train_ds)
        self.assertIsNotNone(val_ds)
        self.assertIsNotNone(test_ds)
        
        # Verify dataset was loaded with correct parameters
        self.mock_datasets.load_dataset.assert_called_once_with("dummy")
    
    def test_data_augmentation(self):
        """Test data augmentation application."""
        from CLIP_HAR_PROJECT.data.augmentation import get_augmentation_transforms
        
        # Get transforms for different strengths
        light_transforms = get_augmentation_transforms(
            image_size=224,
            strength="light"
        )
        medium_transforms = get_augmentation_transforms(
            image_size=224,
            strength="medium"
        )
        strong_transforms = get_augmentation_transforms(
            image_size=224,
            strength="strong"
        )
        
        # Check transformations are created
        self.assertIsNotNone(light_transforms)
        self.assertIsNotNone(medium_transforms)
        self.assertIsNotNone(strong_transforms)
        
        # Verify stronger augmentations have more transforms
        self.assertLess(
            len(light_transforms.transforms), 
            len(strong_transforms.transforms)
        )
    
    def test_class_balance(self):
        """Test class balancing and weighting."""
        from CLIP_HAR_PROJECT.data.dataset import compute_class_weights
        
        # Create mock class distribution
        class_counts = {i: 10 + i * 5 for i in range(15)}  # Unbalanced
        
        # Compute weights
        weights = compute_class_weights(class_counts)
        
        # Check that weights are computed correctly
        self.assertEqual(len(weights), 15)
        
        # Verify less frequent classes have higher weights
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        
        min_count_class = min(class_counts, key=class_counts.get)
        max_count_class = max(class_counts, key=class_counts.get)
        
        self.assertGreater(weights[min_count_class], weights[max_count_class])
    
    def test_preprocessor_transforms(self):
        """Test image and text preprocessing."""
        from CLIP_HAR_PROJECT.data.preprocessing import preprocess_image, tokenize_text
        
        # Create dummy image and text
        dummy_image = np.random.rand(224, 224, 3).astype(np.float32)
        dummy_text = "a photo of a person running"
        
        # Process image
        processed_image = preprocess_image(
            dummy_image, 
            image_processor=self.mock_processor.from_pretrained.return_value
        )
        
        # Tokenize text
        tokenized_text = tokenize_text(
            dummy_text,
            tokenizer=self.mock_tokenizer.from_pretrained.return_value
        )
        
        # Verify preprocessing was called
        self.mock_processor.from_pretrained.return_value.process_image.assert_called_once()
        self.mock_tokenizer.from_pretrained.return_value.encode.assert_called_once()


if __name__ == '__main__':
    unittest.main() 