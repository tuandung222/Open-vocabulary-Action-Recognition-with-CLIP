"""
Inference serving module for CLIP HAR Project.

This module provides utilities for serving CLIP HAR models including:
- Model adapters (PyTorch, ONNX, TorchScript)
- REST API server
- Client for making inference requests
"""

import base64
import io
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
import torch
import torchvision.transforms as T
import uvicorn
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from torch import nn

# Optional imports for model optimization
try:
    import onnx
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

logger = logging.getLogger(__name__)


class InferenceRequest(BaseModel):
    """Model for handling inference requests."""

    image_data: Optional[str] = None
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    top_k: Optional[int] = 5


class InferenceResult(BaseModel):
    """Model for inference results."""

    predictions: List[Dict[str, Any]]
    inference_time: float
    model_name: str


class ModelAdapter:
    """Base class for model adapters."""

    def __init__(
        self,
        model_path: str,
        class_names: Optional[List[str]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the model adapter.

        Args:
            model_path: Path to the model file or directory
            class_names: List of class names
            device: Device to run inference on
        """
        self.model_path = model_path
        self.device = device
        self.class_names = class_names or []
        self.model = None
        self.transform = self._get_transforms()

        # Load model
        self.load_model()

    def _get_transforms(self):
        """Get transforms for preprocessing inputs."""
        # Default transforms for CLIP models
        return T.Compose(
            [
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

    def load_model(self):
        """Load the model."""
        raise NotImplementedError("Subclasses must implement load_model")

    def preprocess(self, image):
        """
        Preprocess the input image.

        Args:
            image: PIL Image or numpy array

        Returns:
            Processed input tensor
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if not isinstance(image, Image.Image):
            raise ValueError(
                f"Input must be PIL Image or numpy array, got {type(image)}"
            )

        return self.transform(image).unsqueeze(0).to(self.device)

    def predict(self, inputs, top_k=5):
        """
        Run inference on inputs.

        Args:
            inputs: Input tensor
            top_k: Number of top predictions to return

        Returns:
            Dictionary of predictions with class names and scores
        """
        raise NotImplementedError("Subclasses must implement predict")

    def predict_from_image(self, image, top_k=5):
        """
        Run inference on a single image.

        Args:
            image: PIL Image or numpy array
            top_k: Number of top predictions to return

        Returns:
            Tuple of (predictions, inference_time)
        """
        start_time = time.time()

        # Preprocess
        inputs = self.preprocess(image)

        # Run inference
        predictions = self.predict(inputs, top_k=top_k)

        inference_time = time.time() - start_time

        return predictions, inference_time


class PyTorchAdapter(ModelAdapter):
    """Adapter for PyTorch models."""

    def load_model(self):
        """Load PyTorch model."""
        try:
            # Handle different types of model files
            if os.path.isdir(self.model_path):
                # Try to load as a Hugging Face model
                from transformers import CLIPModel, CLIPProcessor

                try:
                    self.model = CLIPModel.from_pretrained(self.model_path)
                    self.processor = CLIPProcessor.from_pretrained(self.model_path)
                    self.is_hf_model = True
                    logger.info(f"Loaded HuggingFace CLIP model from {self.model_path}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load as HuggingFace model: {e}")
                    self.is_hf_model = False

            # Try to load as a regular PyTorch model
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if "model" in checkpoint:
                    # Standard format with model state dict
                    model_state = checkpoint["model"]

                    # Try to get model class from checkpoint
                    model_class = checkpoint.get("model_class", None)
                    if model_class is not None:
                        # If the checkpoint contains model class info, use it
                        try:
                            if isinstance(model_class, str):
                                # Import the class dynamically
                                module_path, class_name = model_class.rsplit(".", 1)
                                module = __import__(module_path, fromlist=[class_name])
                                model_class = getattr(module, class_name)

                            # Create model instance
                            self.model = model_class()
                            self.model.load_state_dict(model_state)
                        except Exception as e:
                            logger.warning(
                                f"Failed to instantiate model using class from checkpoint: {e}"
                            )

                            # Fallback to default model
                            from CLIP_HAR_PROJECT.model.clip_classifier import (
                                CLIPClassifier,
                            )

                            self.model = CLIPClassifier()
                            self.model.load_state_dict(model_state)
                    else:
                        # No model class info, use default
                        from CLIP_HAR_PROJECT.model.clip_classifier import (
                            CLIPClassifier,
                        )

                        self.model = CLIPClassifier()
                        self.model.load_state_dict(model_state)

                elif "state_dict" in checkpoint:
                    # PyTorch Lightning format
                    model_state = checkpoint["state_dict"]

                    # Import the default model
                    from CLIP_HAR_PROJECT.model.clip_classifier import CLIPClassifier

                    self.model = CLIPClassifier()

                    # Sometimes lightning adds 'model.' prefix
                    if all(k.startswith("model.") for k in model_state.keys()):
                        # Remove the 'model.' prefix
                        model_state = {k[6:]: v for k, v in model_state.items()}

                    self.model.load_state_dict(model_state)

                elif "model_state_dict" in checkpoint:
                    # Another common format
                    model_state = checkpoint["model_state_dict"]

                    # Import the default model
                    from CLIP_HAR_PROJECT.model.clip_classifier import CLIPClassifier

                    self.model = CLIPClassifier()
                    self.model.load_state_dict(model_state)

                else:
                    # Try loading the whole dict as state dict
                    from CLIP_HAR_PROJECT.model.clip_classifier import CLIPClassifier

                    self.model = CLIPClassifier()
                    self.model.load_state_dict(checkpoint)

            else:
                # Assume checkpoint is the model itself
                self.model = checkpoint

            # Move model to device
            self.model = self.model.to(self.device)

            # Set model to evaluation mode
            self.model.eval()

            # Try to load class names from the checkpoint
            if isinstance(checkpoint, dict) and "class_names" in checkpoint:
                self.class_names = checkpoint["class_names"]

            logger.info(f"Loaded PyTorch model from {self.model_path}")

        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}", exc_info=True)
            raise

    def predict(self, inputs, top_k=5):
        """Run inference with PyTorch model."""
        with torch.no_grad():
            if hasattr(self, "is_hf_model") and self.is_hf_model:
                # For HuggingFace CLIP models
                text_inputs = torch.cat(
                    [
                        self.processor.tokenizer(
                            class_name, return_tensors="pt", padding=True
                        ).to(self.device)["input_ids"]
                        for class_name in self.class_names
                    ]
                )

                image_features = self.model.get_image_features(pixel_values=inputs)
                text_features = self.model.get_text_features(input_ids=text_inputs)

                # Normalize features
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Compute similarity scores
                similarity = 100 * torch.matmul(image_features, text_features.T)

                # Get top predictions
                values, indices = similarity[0].topk(min(top_k, len(self.class_names)))

                predictions = []
                for i, (score, idx) in enumerate(
                    zip(values.cpu().numpy(), indices.cpu().numpy())
                ):
                    predictions.append(
                        {
                            "rank": i + 1,
                            "class_id": int(idx),
                            "class_name": self.class_names[idx]
                            if idx < len(self.class_names)
                            else f"Class {idx}",
                            "score": float(score),
                        }
                    )

                return predictions
            else:
                # For custom models
                outputs = self.model(inputs)

                # Handle different output formats
                if isinstance(outputs, tuple):
                    # Some models return (logits, features)
                    logits = outputs[0]
                else:
                    logits = outputs

                # Get probabilities
                if logits.shape[1] == 1:
                    # Binary classification
                    probs = torch.sigmoid(logits)
                    values, indices = probs.topk(min(top_k, probs.shape[1]))
                else:
                    # Multi-class classification
                    probs = torch.softmax(logits, dim=1)
                    values, indices = probs.topk(min(top_k, probs.shape[1]))

                predictions = []
                for i, (score, idx) in enumerate(
                    zip(values[0].cpu().numpy(), indices[0].cpu().numpy())
                ):
                    predictions.append(
                        {
                            "rank": i + 1,
                            "class_id": int(idx),
                            "class_name": self.class_names[idx]
                            if idx < len(self.class_names)
                            else f"Class {idx}",
                            "score": float(score),
                        }
                    )

                return predictions


class ONNXAdapter(ModelAdapter):
    """Adapter for ONNX models."""

    def __init__(self, *args, **kwargs):
        """Initialize ONNX adapter."""
        if not ONNX_AVAILABLE:
            raise ImportError(
                "ONNX and ONNXRuntime are required for ONNXAdapter. "
                "Install with: pip install onnx onnxruntime"
            )
        super().__init__(*args, **kwargs)

    def load_model(self):
        """Load ONNX model."""
        try:
            # Check if model exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            # Create ONNX session
            if self.device == "cuda" and ort.get_device() == "GPU":
                # Use GPU for inference
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                # Use CPU for inference
                providers = ["CPUExecutionProvider"]

            self.session = ort.InferenceSession(self.model_path, providers=providers)

            # Get model inputs and outputs
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            logger.info(f"Loaded ONNX model from {self.model_path}")

            # Try to load class names from metadata
            try:
                metadata = onnx.load(self.model_path).metadata_props
                for prop in metadata:
                    if prop.key == "class_names":
                        self.class_names = json.loads(prop.value)
                        break
            except Exception as e:
                logger.warning(f"Failed to load class names from ONNX metadata: {e}")

        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}", exc_info=True)
            raise

    def predict(self, inputs, top_k=5):
        """Run inference with ONNX model."""
        # Convert to numpy for ONNX
        inputs_np = inputs.cpu().numpy()

        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: inputs_np})
        logits = outputs[0]

        # Get probabilities
        if logits.shape[1] == 1:
            # Binary classification
            probs = 1 / (1 + np.exp(-logits))  # sigmoid
        else:
            # Multi-class classification
            probs = np.exp(logits) / np.sum(
                np.exp(logits), axis=1, keepdims=True
            )  # softmax

        # Get top predictions
        indices = np.argsort(probs[0])[::-1][:top_k]
        values = probs[0][indices]

        predictions = []
        for i, (idx, score) in enumerate(zip(indices, values)):
            predictions.append(
                {
                    "rank": i + 1,
                    "class_id": int(idx),
                    "class_name": self.class_names[idx]
                    if idx < len(self.class_names)
                    else f"Class {idx}",
                    "score": float(score),
                }
            )

        return predictions


class TorchScriptAdapter(ModelAdapter):
    """Adapter for TorchScript models."""

    def load_model(self):
        """Load TorchScript model."""
        try:
            # Load model
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.model.eval()

            logger.info(f"Loaded TorchScript model from {self.model_path}")

            # Try to load class names
            model_dir = os.path.dirname(self.model_path)
            class_names_file = os.path.join(model_dir, "class_names.json")
            if os.path.exists(class_names_file):
                with open(class_names_file, "r") as f:
                    self.class_names = json.load(f)

        except Exception as e:
            logger.error(f"Failed to load TorchScript model: {e}", exc_info=True)
            raise

    def predict(self, inputs, top_k=5):
        """Run inference with TorchScript model."""
        with torch.no_grad():
            outputs = self.model(inputs)

            # Handle different output formats
            if isinstance(outputs, tuple):
                # Some models return (logits, features)
                logits = outputs[0]
            else:
                logits = outputs

            # Get probabilities
            if logits.shape[1] == 1:
                # Binary classification
                probs = torch.sigmoid(logits)
            else:
                # Multi-class classification
                probs = torch.softmax(logits, dim=1)

            # Get top predictions
            values, indices = probs.topk(min(top_k, probs.shape[1]))

            predictions = []
            for i, (score, idx) in enumerate(
                zip(values[0].cpu().numpy(), indices[0].cpu().numpy())
            ):
                predictions.append(
                    {
                        "rank": i + 1,
                        "class_id": int(idx),
                        "class_name": self.class_names[idx]
                        if idx < len(self.class_names)
                        else f"Class {idx}",
                        "score": float(score),
                    }
                )

            return predictions


class InferenceService:
    """Service for running inference with different model adapters."""

    def __init__(
        self,
        model_path: str,
        model_type: str = "pytorch",
        class_names: Optional[List[str]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the inference service.

        Args:
            model_path: Path to the model file or directory
            model_type: Type of model (pytorch, onnx, torchscript)
            class_names: List of class names
            device: Device to run inference on
        """
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.class_names = class_names
        self.device = device
        self.model_name = os.path.basename(model_path)

        # Load model adapter
        self._load_adapter()

    def _load_adapter(self):
        """Load the appropriate model adapter."""
        if self.model_type == "pytorch":
            self.adapter = PyTorchAdapter(
                model_path=self.model_path,
                class_names=self.class_names,
                device=self.device,
            )
        elif self.model_type == "onnx":
            self.adapter = ONNXAdapter(
                model_path=self.model_path,
                class_names=self.class_names,
                device=self.device,
            )
        elif self.model_type == "torchscript":
            self.adapter = TorchScriptAdapter(
                model_path=self.model_path,
                class_names=self.class_names,
                device=self.device,
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def predict_from_image(self, image, top_k=5) -> Tuple[List[Dict[str, Any]], float]:
        """
        Run inference on an image.

        Args:
            image: PIL Image or numpy array
            top_k: Number of top predictions to return

        Returns:
            Tuple of (predictions, inference_time)
        """
        return self.adapter.predict_from_image(image, top_k=top_k)

    def predict_from_image_bytes(
        self, image_bytes, top_k=5
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Run inference on image bytes.

        Args:
            image_bytes: Bytes of the image
            top_k: Number of top predictions to return

        Returns:
            Tuple of (predictions, inference_time)
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.predict_from_image(image, top_k=top_k)

    def predict_from_image_base64(
        self, image_base64, top_k=5
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Run inference on base64 encoded image.

        Args:
            image_base64: Base64 encoded image string
            top_k: Number of top predictions to return

        Returns:
            Tuple of (predictions, inference_time)
        """
        image_bytes = base64.b64decode(image_base64)
        return self.predict_from_image_bytes(image_bytes, top_k=top_k)

    def predict_from_image_url(
        self, image_url, top_k=5
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Run inference on image from URL.

        Args:
            image_url: URL of the image
            top_k: Number of top predictions to return

        Returns:
            Tuple of (predictions, inference_time)
        """
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        return self.predict_from_image(image, top_k=top_k)


def create_inference_app(
    model_path: str,
    model_type: str = "pytorch",
    class_names: Optional[List[str]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> FastAPI:
    """
    Create a FastAPI application for serving inference.

    Args:
        model_path: Path to the model file or directory
        model_type: Type of model (pytorch, onnx, torchscript)
        class_names: List of class names
        device: Device to run inference on

    Returns:
        FastAPI application
    """
    # Initialize service
    service = InferenceService(
        model_path=model_path,
        model_type=model_type,
        class_names=class_names,
        device=device,
    )

    # Create app
    app = FastAPI(title="CLIP HAR Inference Service", version="1.0.0")

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        """Get service information."""
        return {
            "name": "CLIP HAR Inference Service",
            "model_name": service.model_name,
            "model_type": service.model_type,
            "device": service.device,
            "class_names": service.adapter.class_names,
        }

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.post("/predict")
    async def predict(request: InferenceRequest):
        """Run inference on an image."""
        try:
            # Check inputs
            if request.image_data:
                # Process base64 image
                predictions, inference_time = service.predict_from_image_base64(
                    request.image_data, top_k=request.top_k or 5
                )
            elif request.image_url:
                # Process image URL
                predictions, inference_time = service.predict_from_image_url(
                    request.image_url, top_k=request.top_k or 5
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="No input provided. Please provide either image_data or image_url.",
                )

            return InferenceResult(
                predictions=predictions,
                inference_time=inference_time,
                model_name=service.model_name,
            )

        except Exception as e:
            logger.error(f"Error during inference: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Error during inference: {str(e)}"
            )

    @app.post("/predict/image")
    async def predict_image(file: UploadFile = File(...), top_k: int = Form(5)):
        """Run inference on an uploaded image."""
        try:
            # Read image
            image_bytes = await file.read()

            # Run inference
            predictions, inference_time = service.predict_from_image_bytes(
                image_bytes, top_k=top_k
            )

            return InferenceResult(
                predictions=predictions,
                inference_time=inference_time,
                model_name=service.model_name,
            )

        except Exception as e:
            logger.error(f"Error during inference: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Error during inference: {str(e)}"
            )

    return app


def serve_model(
    model_path: str,
    model_type: str = "pytorch",
    class_names: Optional[List[str]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    host: str = "0.0.0.0",
    port: int = 8000,
):
    """
    Serve a model for inference.

    Args:
        model_path: Path to the model file or directory
        model_type: Type of model (pytorch, onnx, torchscript)
        class_names: List of class names
        device: Device to run inference on
        host: Host to serve on
        port: Port to serve on
    """
    # Create app
    app = create_inference_app(
        model_path=model_path,
        model_type=model_type,
        class_names=class_names,
        device=device,
    )

    # Run server
    uvicorn.run(app, host=host, port=port)


class InferenceClient:
    """Client for making inference requests to the inference service."""

    def __init__(self, url: str = "http://localhost:8000"):
        """
        Initialize the client.

        Args:
            url: URL of the inference service
        """
        self.url = url.rstrip("/")

    def predict_from_image(self, image, top_k=5) -> Dict[str, Any]:
        """
        Run inference on an image.

        Args:
            image: PIL Image
            top_k: Number of top predictions to return

        Returns:
            Dictionary with predictions and inference time
        """
        # Convert to bytes
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)

        # Send request
        files = {"file": ("image.jpg", buffer, "image/jpeg")}
        data = {"top_k": top_k}

        response = requests.post(f"{self.url}/predict/image", files=files, data=data)
        response.raise_for_status()

        return response.json()

    def predict_from_image_path(self, image_path, top_k=5) -> Dict[str, Any]:
        """
        Run inference on an image from file path.

        Args:
            image_path: Path to the image file
            top_k: Number of top predictions to return

        Returns:
            Dictionary with predictions and inference time
        """
        image = Image.open(image_path).convert("RGB")
        return self.predict_from_image(image, top_k=top_k)

    def predict_from_image_url(self, image_url, top_k=5) -> Dict[str, Any]:
        """
        Run inference on an image from URL.

        Args:
            image_url: URL of the image
            top_k: Number of top predictions to return

        Returns:
            Dictionary with predictions and inference time
        """
        data = {"image_url": image_url, "top_k": top_k}

        response = requests.post(f"{self.url}/predict", json=data)
        response.raise_for_status()

        return response.json()

    def predict_from_image_base64(self, image_base64, top_k=5) -> Dict[str, Any]:
        """
        Run inference on a base64 encoded image.

        Args:
            image_base64: Base64 encoded image string
            top_k: Number of top predictions to return

        Returns:
            Dictionary with predictions and inference time
        """
        data = {"image_data": image_base64, "top_k": top_k}

        response = requests.post(f"{self.url}/predict", json=data)
        response.raise_for_status()

        return response.json()


def main():
    """Command line entrypoint for serving a model."""
    import argparse

    parser = argparse.ArgumentParser(description="Serve a CLIP HAR model for inference")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model file or directory",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="pytorch",
        choices=["pytorch", "onnx", "torchscript"],
        help="Type of model",
    )
    parser.add_argument(
        "--class_names",
        type=str,
        default=None,
        help="Path to JSON file with class names",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to serve on")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")

    args = parser.parse_args()

    # Load class names if provided
    class_names = None
    if args.class_names:
        with open(args.class_names, "r") as f:
            class_names = json.load(f)

    # Serve model
    serve_model(
        model_path=args.model_path,
        model_type=args.model_type,
        class_names=class_names,
        device=args.device,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
