import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizerFast, CLIPImageProcessor
from typing import Dict, List, Optional, Union, Any, Tuple


class CLIPLabelRetriever(nn.Module):
    """
    A class that uses CLIP to classify images based on their similarity to text prompts.
    """

    def __init__(
        self,
        clip_model: CLIPModel,
        tokenizer: CLIPTokenizerFast,
        labels: List[str],
        prompt_template: str = "a photo of person/people who is/are {label}",
    ):
        super().__init__()
        # Verify CLIP model has necessary methods
        assert hasattr(clip_model, "get_text_features")
        assert hasattr(clip_model, "get_image_features")
        assert hasattr(clip_model, "logit_scale")

        self.model = clip_model
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.label2id = {label: i for i, label in enumerate(labels)}
        self.id2label = {i: label for i, label in enumerate(labels)}
        self.labels = labels
        self._cached_text_features = None

    def _get_device(self):
        """Get the device of the model."""
        return next(self.model.parameters()).device

    def generate_labels_embeddings(self):
        """
        Function prepares label's prompt embeddings using the language model.

        Returns:
            torch.Tensor: labels embeddings of size (num_labels, dim)
        """
        if self._cached_text_features is not None:
            return self._cached_text_features

        # Generate prompts from labels
        prompts = [self.prompt_template.format(label=label) for label in self.labels]

        # Tokenize label's prompt
        labels_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        ).to(self._get_device())

        # Run language model on this prompt and get embeddings
        with torch.no_grad():
            labels_embeddings = self.model.get_text_features(**labels_inputs)

        # Normalize embeddings
        norm_labels_embeddings = labels_embeddings / labels_embeddings.norm(
            p=2, dim=-1, keepdim=True
        )

        # Cache the text features to avoid recomputation
        self._cached_text_features = norm_labels_embeddings

        return norm_labels_embeddings

    def clear_cache(self):
        """Clear the cached text features."""
        self._cached_text_features = None

    def forward(self, return_loss: bool = True, **batch):
        """
        Calculate similarity between image and label's prompts embeddings.

        Args:
            batch: Batch of images and labels
            return_loss: Whether to return loss or logits

        Returns:
            Loss if return_loss=True, otherwise similarity scores
        """
        # Get text features for all labels
        labels_embeddings = self.generate_labels_embeddings()

        # Get image features
        images = batch["pixel_values"]
        image_features = self.model.get_image_features(images)  # (batch_size, dim)

        # Normalize image features
        norm_image_features = image_features / image_features.norm(
            p=2, dim=-1, keepdim=True
        )  # (batch_size, dim)

        # Calculate similarity scores
        score_tensor = (
            torch.matmul(norm_image_features, labels_embeddings.T)
            * self.model.logit_scale.exp()
        )

        # Return loss or similarity scores
        if "label_id" in batch and return_loss:
            labels = batch["label_id"]
            loss = F.cross_entropy(score_tensor, labels)
            return loss
        else:
            return score_tensor

    def predict(self, pixel_values):
        """
        Predict the class of the given images.

        Args:
            pixel_values: Image tensors of shape (B, 3, H, W)

        Returns:
            Tuple of (predicted class ids, predicted class names, similarity scores)
        """
        # Get similarity scores
        batch = {"pixel_values": pixel_values}
        with torch.no_grad():
            similarity_scores = self.forward(return_loss=False, **batch)

        # Get predictions
        predicted_class_ids = torch.argmax(similarity_scores, dim=1)
        predicted_class_names = [self.id2label[int(i)] for i in predicted_class_ids]

        return predicted_class_ids, predicted_class_names, similarity_scores

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        labels: List[str],
        prompt_template: str = "a photo of person/people who is/are {label}",
    ):
        """
        Create a CLIPLabelRetriever from a pretrained CLIP model.

        Args:
            model_name_or_path: The name or path of the pretrained model
            labels: The list of labels
            prompt_template: The template for creating prompts from labels

        Returns:
            A CLIPLabelRetriever instance
        """
        tokenizer = CLIPTokenizerFast.from_pretrained(model_name_or_path)
        model = CLIPModel.from_pretrained(model_name_or_path)

        return cls(model, tokenizer, labels, prompt_template)


def freeze_clip_parameters(
    model: nn.Module,
    unfreeze_visual_encoder: bool = False,
    unfreeze_text_encoder: bool = False,
):
    """
    Freeze parameters of the CLIP model.

    Args:
        model: The CLIP model or CLIPLabelRetriever
        unfreeze_visual_encoder: Whether to unfreeze the visual encoder
        unfreeze_text_encoder: Whether to unfreeze the text encoder
    """
    # Get the CLIP model
    clip_model = model.model if isinstance(model, CLIPLabelRetriever) else model

    # Freeze all parameters by default
    for param in clip_model.parameters():
        param.requires_grad = False

    # Unfreeze visual encoder if specified
    if unfreeze_visual_encoder:
        for param in clip_model.vision_model.parameters():
            param.requires_grad = True

    # Unfreeze text encoder if specified
    if unfreeze_text_encoder:
        for param in clip_model.text_model.parameters():
            param.requires_grad = True

    # Always unfreeze the logit scale parameter
    if hasattr(clip_model, "logit_scale"):
        clip_model.logit_scale.requires_grad = True


def print_trainable_parameters(model: nn.Module):
    """
    Print the number of trainable parameters in the model.

    Args:
        model: The model
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"Trainable params: {(trainable_params / 10**6):.4f}M || "
        f"All params: {(all_param / 10**6):.4f}M || "
        f"Trainable%: {100 * trainable_params / all_param:.2f}%"
    )


def get_labels_from_dataset(dataset):
    """
    Extract labels from the dataset.

    Args:
        dataset: A dataset with labels

    Returns:
        A list of unique labels
    """
    if hasattr(dataset, "features") and "labels" in dataset.features:
        label_names = dataset.features["labels"].names
        return label_names

    # Extract unique labels from the dataset
    unique_labels = set()
    for example in dataset:
        if "labels" in example:
            unique_labels.add(example["labels"])
        elif "label" in example:
            unique_labels.add(example["label"])

    return sorted(list(unique_labels))
