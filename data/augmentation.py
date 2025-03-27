import random
import numpy as np
import torch
import torchvision.transforms as T
from typing import Dict, List, Optional, Tuple, Union, Callable


class HARDataAugmentation:
    """
    Data augmentation techniques for Human Action Recognition.

    This class provides various augmentation pipelines suitable for HAR tasks,
    with configurations that preserve action-specific visual cues.
    """

    @staticmethod
    def get_train_transforms(
        image_size: int = 224,
        mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073),
        std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711),
        augmentation_strength: str = "medium",
    ) -> Callable:
        """
        Get training transforms with augmentations.

        Args:
            image_size: Target image size
            mean: Normalization mean values
            std: Normalization standard deviation values
            augmentation_strength: Level of augmentation ("light", "medium", "strong")

        Returns:
            Composition of transforms
        """
        base_transforms = [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]

        if augmentation_strength == "light":
            augmentations = [
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ]
        elif augmentation_strength == "medium":
            augmentations = [
                T.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.RandomGrayscale(p=0.02),
            ]
        elif augmentation_strength == "strong":
            augmentations = [
                T.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                T.RandomGrayscale(p=0.05),
                T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                T.RandomPerspective(distortion_scale=0.2, p=0.3),
            ]
        else:
            raise ValueError(f"Unknown augmentation strength: {augmentation_strength}")

        return T.Compose(augmentations + base_transforms)

    @staticmethod
    def get_eval_transforms(
        image_size: int = 224,
        mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073),
        std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711),
    ) -> Callable:
        """
        Get evaluation transforms without augmentations.

        Args:
            image_size: Target image size
            mean: Normalization mean values
            std: Normalization standard deviation values

        Returns:
            Composition of transforms
        """
        return T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    @staticmethod
    def get_action_specific_transforms(
        action_name: str,
        image_size: int = 224,
        mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073),
        std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711),
    ) -> Callable:
        """
        Get action-specific transforms tailored to particular actions.

        Args:
            action_name: Name of the action class
            image_size: Target image size
            mean: Normalization mean values
            std: Normalization standard deviation values

        Returns:
            Composition of transforms
        """
        base_transforms = [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]

        if action_name in ["running", "cycling", "dancing"]:
            # For active motions: more aggressive cropping and motion blur
            action_transforms = [
                T.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                # Optional: Add motion blur in custom transform
            ]
        elif action_name in ["drinking", "eating", "texting"]:
            # For fine-grained actions: focus on detail preservation
            action_transforms = [
                T.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
                T.RandomHorizontalFlip(p=0.3),
                T.ColorJitter(brightness=0.1, contrast=0.1),
            ]
        elif action_name in ["fighting", "hugging"]:
            # For interaction actions: preserve spatial relationships
            action_transforms = [
                T.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        else:
            # Default transforms for other actions
            action_transforms = [
                T.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]

        return T.Compose(action_transforms + base_transforms)

    @staticmethod
    def get_mix_augmentation(
        mix_type: str = "mixup", alpha: float = 0.2, cutmix_prob: float = 0.5
    ) -> Dict:
        """
        Get configuration for mixup or cutmix augmentations.

        Args:
            mix_type: Type of mix augmentation ("mixup", "cutmix", or "both")
            alpha: Alpha parameter for Beta distribution
            cutmix_prob: Probability of applying cutmix when mix_type is "both"

        Returns:
            Dictionary with mix augmentation configuration
        """
        config = {
            "enabled": True,
            "alpha": alpha,
        }

        if mix_type == "mixup":
            config["mode"] = "mixup"
        elif mix_type == "cutmix":
            config["mode"] = "cutmix"
        elif mix_type == "both":
            config["mode"] = "both"
            config["cutmix_prob"] = cutmix_prob
        else:
            config["enabled"] = False

        return config


def apply_mixup(
    images: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Apply mixup augmentation to a batch of images.

    Args:
        images: Batch of images (B, C, H, W)
        labels: One-hot encoded labels (B, num_classes)
        alpha: Mixup strength parameter

    Returns:
        Tuple of (mixed_images, mixed_labels, lambda)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1 - lam) * images[index]
    mixed_labels = lam * labels + (1 - lam) * labels[index]

    return mixed_images, mixed_labels, lam


def apply_cutmix(
    images: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Apply cutmix augmentation to a batch of images.

    Args:
        images: Batch of images (B, C, H, W)
        labels: One-hot encoded labels (B, num_classes)
        alpha: CutMix strength parameter

    Returns:
        Tuple of (mixed_images, mixed_labels, lambda)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    h, w = images.size()[-2:]
    center_h = int(random.uniform(0, h))
    center_w = int(random.uniform(0, w))

    # Calculate box size
    half_box_h = int(np.sqrt(1.0 - lam) * h * 0.5)
    half_box_w = int(np.sqrt(1.0 - lam) * w * 0.5)

    # Calculate box coordinates
    x1 = max(0, center_w - half_box_w)
    y1 = max(0, center_h - half_box_h)
    x2 = min(w, center_w + half_box_w)
    y2 = min(h, center_h + half_box_h)

    # Mix images
    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

    # Adjust lambda to match the box area
    box_area = (y2 - y1) * (x2 - x1)
    lam = 1 - box_area / (h * w)

    # Mix labels
    mixed_labels = lam * labels + (1 - lam) * labels[index]

    return mixed_images, mixed_labels, lam
