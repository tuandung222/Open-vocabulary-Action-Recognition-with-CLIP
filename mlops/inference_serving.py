"""
Model inference serving module for the CLIP HAR project.

This module provides a FastAPI-based model serving API with support
for multiple model formats including PyTorch, ONNX, and TensorRT.
"""

import os
import io
import base64
import json
import logging
import uvicorn
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
from abc import ABC, abstractmethod
from PIL import Image
import requests
from pydantic import BaseModel, Field
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if ONNX Runtime is available
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    logger.warning("ONNX Runtime not available. ONNX models will not be supported.")
    ONNX_AVAILABLE = False

# Check if TensorRT is available
try:
    import tensorrt as trt
    from CLIP_HAR_PROJECT.deployment.optimization import TensorRTOptimizer
    TENSORRT_AVAILABLE = True
except ImportError:
    logger.warning("TensorRT not available. TensorRT models will not be supported.")
    TENSORRT_AVAILABLE = False


class ModelType(str, Enum):
    """Supported model types for inference."""
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    TENSORRT = "tensorrt"


class ImageInput(BaseModel):
    """Input for image-based prediction."""
    image_data: Optional[str] = Field(None, description="Base64 encoded image data")
    image_url: Optional[str] = Field(None, description="URL to an image")
    
    class Config:
        schema_extra = {
            "example": {
                "image_data": "base64_encoded_image_data_here",
                "image_url": "https://example.com/image.jpg"
            }
        }


class PredictionOutput(BaseModel):
    """Output for prediction requests."""
    predictions: List[Dict[str, Any]] = Field(
        ..., description="List of predictions with class name and score"
    )
    model_info: Dict[str, Any] = Field(
        ..., description="Information about the model used for inference"
    )
    inference_time_ms: float = Field(
        ..., description="Time taken for inference in milliseconds"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {"class_name": "running", "score": 0.92},
                    {"class_name": "walking", "score": 0.05},
                    {"class_name": "jumping", "score": 0.03}
                ],
                "model_info": {
                    "model_type": "onnx",
                    "model_name": "clip_har_v1"
                },
                "inference_time_ms": 15.5
            }
        }


class ModelAdapter(ABC):
    """Abstract base class for model adapters."""
    
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """
        Load a model from a file.
        
        Args:
            model_path: Path to the model file
        """
        pass
    
    @abstractmethod
    def preprocess(self, image: Image.Image) -> Any:
        """
        Preprocess an image for model input.
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Preprocessed image in the format expected by the model
        """
        pass
    
    @abstractmethod
    def predict(self, inputs: Any) -> np.ndarray:
        """
        Run inference on the preprocessed inputs.
        
        Args:
            inputs: Preprocessed inputs
            
        Returns:
            Model predictions as a numpy array
        """
        pass
    
    @abstractmethod
    def postprocess(self, outputs: np.ndarray, class_names: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Postprocess model outputs to human-readable predictions.
        
        Args:
            outputs: Model output
            class_names: List of class names
            top_k: Number of top predictions to return
            
        Returns:
            List of dictionaries with class names and scores
        """
        pass


class PyTorchAdapter(ModelAdapter):
    """Adapter for PyTorch models."""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the PyTorch adapter.
        
        Args:
            device: Device to run inference on
        """
        self.device = torch.device(device)
        self.model = None
        self.input_size = (224, 224)
        self.mean = [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]
    
    def load_model(self, model_path: str) -> None:
        """
        Load a PyTorch model from a file.
        
        Args:
            model_path: Path to the model file
        """
        try:
            logger.info(f"Loading PyTorch model from {model_path}")
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()
            logger.info("PyTorch model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            raise RuntimeError(f"Failed to load PyTorch model: {e}")
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess an image for model input.
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Preprocessed image as a PyTorch tensor
        """
        from torchvision import transforms
        
        # Resize and normalize image
        transform = transforms.Compose([
            transforms.Resize(self.input_size, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        # Apply transformations
        tensor = transform(image).unsqueeze(0).to(self.device)
        return tensor
    
    def predict(self, inputs: torch.Tensor) -> np.ndarray:
        """
        Run inference on the preprocessed inputs.
        
        Args:
            inputs: Preprocessed inputs as a PyTorch tensor
            
        Returns:
            Model predictions as a numpy array
        """
        with torch.no_grad():
            outputs = self.model(inputs)
            
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            return probs.cpu().numpy()
    
    def postprocess(
        self, outputs: np.ndarray, class_names: List[str], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Postprocess model outputs to human-readable predictions.
        
        Args:
            outputs: Model output as a numpy array
            class_names: List of class names
            top_k: Number of top predictions to return
            
        Returns:
            List of dictionaries with class names and scores
        """
        # Get top-k predictions
        top_indices = np.argsort(outputs[0])[::-1][:top_k]
        
        # Create predictions list
        predictions = [
            {
                "class_name": class_names[idx],
                "score": float(outputs[0][idx])
            }
            for idx in top_indices
        ]
        
        return predictions


class ONNXAdapter(ModelAdapter):
    """Adapter for ONNX models."""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the ONNX adapter.
        
        Args:
            device: Device to run inference on
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime is not available. Please install onnxruntime or onnxruntime-gpu.")
        
        self.device = device
        
        # Configure ONNX runtime session
        if self.device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            self.providers = ["CPUExecutionProvider"]
            
        logger.info(f"Using ONNX Runtime with providers: {self.providers}")
        
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_size = (224, 224)
        self.mean = [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]
    
    def load_model(self, model_path: str) -> None:
        """
        Load an ONNX model from a file.
        
        Args:
            model_path: Path to the model file
        """
        try:
            logger.info(f"Loading ONNX model from {model_path}")
            self.session = ort.InferenceSession(model_path, providers=self.providers)
            
            # Get input and output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            logger.info(f"ONNX model loaded successfully with input: {self.input_name}, output: {self.output_name}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise RuntimeError(f"Failed to load ONNX model: {e}")
    
    def preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess an image for model input.
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Preprocessed image as a numpy array
        """
        # Resize image
        image = image.resize(self.input_size)
        
        # Convert to numpy array
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Apply normalization
        for i in range(3):
            img_array[:, :, i] = (img_array[:, :, i] - self.mean[i]) / self.std[i]
        
        # Rearrange from HWC to CHW
        img_array = img_array.transpose(2, 0, 1)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run inference on the preprocessed inputs.
        
        Args:
            inputs: Preprocessed inputs as a numpy array
            
        Returns:
            Model predictions as a numpy array
        """
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: inputs})
        
        # Apply softmax to get probabilities
        probs = self._softmax(outputs[0])
        
        return probs
    
    def postprocess(
        self, outputs: np.ndarray, class_names: List[str], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Postprocess model outputs to human-readable predictions.
        
        Args:
            outputs: Model output as a numpy array
            class_names: List of class names
            top_k: Number of top predictions to return
            
        Returns:
            List of dictionaries with class names and scores
        """
        # Get top-k predictions
        top_indices = np.argsort(outputs[0])[::-1][:top_k]
        
        # Create predictions list
        predictions = [
            {
                "class_name": class_names[idx],
                "score": float(outputs[0][idx])
            }
            for idx in top_indices
        ]
        
        return predictions
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax to an array."""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)


class TorchScriptAdapter(ModelAdapter):
    """Adapter for TorchScript models."""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the TorchScript adapter.
        
        Args:
            device: Device to run inference on
        """
        self.device = torch.device(device)
        self.model = None
        self.input_size = (224, 224)
        self.mean = [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]
    
    def load_model(self, model_path: str) -> None:
        """
        Load a TorchScript model from a file.
        
        Args:
            model_path: Path to the model file
        """
        try:
            logger.info(f"Loading TorchScript model from {model_path}")
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            logger.info("TorchScript model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load TorchScript model: {e}")
            raise RuntimeError(f"Failed to load TorchScript model: {e}")
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess an image for model input.
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Preprocessed image as a PyTorch tensor
        """
        from torchvision import transforms
        
        # Resize and normalize image
        transform = transforms.Compose([
            transforms.Resize(self.input_size, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        # Apply transformations
        tensor = transform(image).unsqueeze(0).to(self.device)
        return tensor
    
    def predict(self, inputs: torch.Tensor) -> np.ndarray:
        """
        Run inference on the preprocessed inputs.
        
        Args:
            inputs: Preprocessed inputs as a PyTorch tensor
            
        Returns:
            Model predictions as a numpy array
        """
        with torch.no_grad():
            outputs = self.model(inputs)
            
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            return probs.cpu().numpy()
    
    def postprocess(
        self, outputs: np.ndarray, class_names: List[str], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Postprocess model outputs to human-readable predictions.
        
        Args:
            outputs: Model output as a numpy array
            class_names: List of class names
            top_k: Number of top predictions to return
            
        Returns:
            List of dictionaries with class names and scores
        """
        # Get top-k predictions
        top_indices = np.argsort(outputs[0])[::-1][:top_k]
        
        # Create predictions list
        predictions = [
            {
                "class_name": class_names[idx],
                "score": float(outputs[0][idx])
            }
            for idx in top_indices
        ]
        
        return predictions


class TensorRTAdapter(ModelAdapter):
    """Adapter for TensorRT models."""
    
    def __init__(self):
        """Initialize the TensorRT adapter."""
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT is not available. Please install TensorRT and PyCUDA.")
        
        self.optimizer = None
        self.input_name = None
        self.input_size = (224, 224)
        self.mean = [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]
    
    def load_model(self, model_path: str) -> None:
        """
        Load a TensorRT engine from a file.
        
        Args:
            model_path: Path to the TensorRT engine file
        """
        try:
            logger.info(f"Loading TensorRT engine from {model_path}")
            self.optimizer = TensorRTOptimizer(max_batch_size=1)
            self.optimizer.load_engine(model_path)
            self.optimizer.prepare_inference()
            
            # Assuming the first binding is the input
            for binding_idx in range(self.optimizer.engine.num_bindings):
                if self.optimizer.engine.binding_is_input(binding_idx):
                    self.input_name = self.optimizer.engine.get_binding_name(binding_idx)
                    break
            
            logger.info(f"TensorRT engine loaded successfully with input: {self.input_name}")
        except Exception as e:
            logger.error(f"Failed to load TensorRT engine: {e}")
            raise RuntimeError(f"Failed to load TensorRT engine: {e}")
    
    def preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess an image for model input.
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Preprocessed image as a numpy array
        """
        # Resize image
        image = image.resize(self.input_size)
        
        # Convert to numpy array
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Apply normalization
        for i in range(3):
            img_array[:, :, i] = (img_array[:, :, i] - self.mean[i]) / self.std[i]
        
        # Rearrange from HWC to CHW
        img_array = img_array.transpose(2, 0, 1)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run inference on the preprocessed inputs.
        
        Args:
            inputs: Preprocessed inputs as a numpy array
            
        Returns:
            Model predictions as a numpy array
        """
        # Prepare inputs dictionary
        input_dict = {self.input_name: inputs}
        
        # Run inference
        outputs = self.optimizer.infer(input_dict)
        
        # Get the first output (assuming there's only one)
        output_name = list(outputs.keys())[0]
        output_array = outputs[output_name]
        
        # Apply softmax to get probabilities
        probs = self._softmax(output_array)
        
        return probs
    
    def postprocess(
        self, outputs: np.ndarray, class_names: List[str], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Postprocess model outputs to human-readable predictions.
        
        Args:
            outputs: Model output as a numpy array
            class_names: List of class names
            top_k: Number of top predictions to return
            
        Returns:
            List of dictionaries with class names and scores
        """
        # Get top-k predictions
        top_indices = np.argsort(outputs[0])[::-1][:top_k]
        
        # Create predictions list
        predictions = [
            {
                "class_name": class_names[idx],
                "score": float(outputs[0][idx])
            }
            for idx in top_indices
        ]
        
        return predictions
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax to an array."""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)


class ModelServer:
    """Server for model inference."""
    
    def __init__(
        self,
        model_path: str,
        model_type: ModelType,
        class_names: Optional[List[str]] = None,
        class_names_path: Optional[str] = None,
        top_k: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the model server.
        
        Args:
            model_path: Path to the model file
            model_type: Type of the model
            class_names: List of class names
            class_names_path: Path to a JSON file with class names
            top_k: Number of top predictions to return
            device: Device to run inference on
        """
        self.model_path = model_path
        self.model_type = model_type
        self.top_k = top_k
        self.device = device
        
        # Load class names
        if class_names:
            self.class_names = class_names
        elif class_names_path:
            try:
                with open(class_names_path, "r") as f:
                    self.class_names = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load class names from {class_names_path}: {e}")
                raise ValueError(f"Failed to load class names: {e}")
        else:
            logger.warning("No class names provided. Using indices as class names.")
            self.class_names = [str(i) for i in range(1000)]  # Default to 1000 classes
        
        # Create model adapter based on model type
        if model_type == ModelType.PYTORCH:
            self.adapter = PyTorchAdapter(device=device)
        elif model_type == ModelType.ONNX:
            if not ONNX_AVAILABLE:
                raise ImportError("ONNX Runtime is not available. Please install onnxruntime or onnxruntime-gpu.")
            self.adapter = ONNXAdapter(device=device)
        elif model_type == ModelType.TORCHSCRIPT:
            self.adapter = TorchScriptAdapter(device=device)
        elif model_type == ModelType.TENSORRT:
            if not TENSORRT_AVAILABLE:
                raise ImportError("TensorRT is not available. Please install TensorRT and PyCUDA.")
            self.adapter = TensorRTAdapter()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Load the model
        self.adapter.load_model(model_path)
        
        logger.info(f"Model server initialized with {model_type} model at {model_path}")
    
    def predict_from_image(self, image: Image.Image) -> Tuple[List[Dict[str, Any]], float]:
        """
        Run inference on an image.
        
        Args:
            image: PIL Image to run inference on
            
        Returns:
            Tuple of (predictions, inference_time_ms)
        """
        import time
        
        # Preprocess image
        inputs = self.adapter.preprocess(image)
        
        # Run inference and measure time
        start_time = time.time()
        outputs = self.adapter.predict(inputs)
        end_time = time.time()
        
        # Calculate inference time in milliseconds
        inference_time_ms = (end_time - start_time) * 1000
        
        # Postprocess outputs
        predictions = self.adapter.postprocess(outputs, self.class_names, self.top_k)
        
        return predictions, inference_time_ms


class InferenceClient:
    """Client for the inference API."""
    
    def __init__(self, url: str = "http://localhost:8000"):
        """
        Initialize the inference client.
        
        Args:
            url: URL of the inference API
        """
        self.url = url.rstrip("/")
    
    def predict_from_image_path(self, image_path: str) -> Dict[str, Any]:
        """
        Run inference on an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Prediction results
        """
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
            response = requests.post(f"{self.url}/predict/image", files=files)
        
        return self._handle_response(response)
    
    def predict_from_image_url(self, image_url: str) -> Dict[str, Any]:
        """
        Run inference on an image URL.
        
        Args:
            image_url: URL to the image
            
        Returns:
            Prediction results
        """
        data = {"image_url": image_url}
        response = requests.post(f"{self.url}/predict", json=data)
        
        return self._handle_response(response)
    
    def predict_from_image_base64(self, image_base64: str) -> Dict[str, Any]:
        """
        Run inference on a base64-encoded image.
        
        Args:
            image_base64: Base64-encoded image data
            
        Returns:
            Prediction results
        """
        data = {"image_data": image_base64}
        response = requests.post(f"{self.url}/predict", json=data)
        
        return self._handle_response(response)
    
    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Handle API response.
        
        Args:
            response: API response
            
        Returns:
            Response JSON data
        """
        if response.status_code != 200:
            raise RuntimeError(f"API request failed with status code {response.status_code}: {response.text}")
        
        return response.json()


def create_app(
    model_path: str,
    model_type: str,
    class_names: Optional[List[str]] = None,
    class_names_path: Optional[str] = None,
    top_k: int = 5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> FastAPI:
    """
    Create a FastAPI application for model inference.
    
    Args:
        model_path: Path to the model file
        model_type: Type of the model
        class_names: List of class names
        class_names_path: Path to a JSON file with class names
        top_k: Number of top predictions to return
        device: Device to run inference on
        
    Returns:
        FastAPI application
    """
    # Create FastAPI app
    app = FastAPI(
        title="CLIP HAR Inference API",
        description="API for Human Action Recognition using CLIP",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )
    
    # Create model server
    try:
        model_server = ModelServer(
            model_path=model_path,
            model_type=ModelType(model_type),
            class_names=class_names,
            class_names_path=class_names_path,
            top_k=top_k,
            device=device
        )
    except Exception as e:
        logger.error(f"Failed to initialize model server: {e}")
        raise RuntimeError(f"Failed to initialize model server: {e}")
    
    # Store model info
    model_info = {
        "model_type": model_type,
        "model_path": model_path,
        "device": device,
        "num_classes": len(model_server.class_names),
        "class_names": model_server.class_names[:10] + ["..."] if len(model_server.class_names) > 10 else model_server.class_names,
        "top_k": top_k
    }
    
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "CLIP HAR Inference API",
            "version": "1.0.0",
            "model_info": model_info,
            "endpoints": {
                "/": "API information",
                "/health": "Health check",
                "/predict": "Run inference on an image",
                "/predict/image": "Run inference on an uploaded image"
            }
        }
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "ok", "model": model_info}
    
    async def _predict_image(image: Image.Image) -> Dict[str, Any]:
        """
        Run inference on an image.
        
        Args:
            image: PIL Image to run inference on
            
        Returns:
            Prediction results
        """
        try:
            predictions, inference_time_ms = model_server.predict_from_image(image)
            
            result = PredictionOutput(
                predictions=predictions,
                model_info={
                    "model_type": model_type,
                    "model_name": os.path.basename(model_path)
                },
                inference_time_ms=inference_time_ms
            )
            
            return result.dict()
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    
    @app.post("/predict", response_model=PredictionOutput)
    async def predict(input_data: ImageInput):
        """
        Run inference on an image.
        
        Args:
            input_data: Input data with image data or URL
            
        Returns:
            Prediction results
        """
        # Check if input is valid
        if not input_data.image_data and not input_data.image_url:
            raise HTTPException(
                status_code=400,
                detail="Either image_data or image_url must be provided"
            )
        
        try:
            # Load image from base64 data
            if input_data.image_data:
                image_data = base64.b64decode(input_data.image_data)
                image = Image.open(io.BytesIO(image_data))
            # Load image from URL
            elif input_data.image_url:
                response = requests.get(input_data.image_url, stream=True)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content))
            
            # Run inference
            return await _predict_image(image)
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch image from URL: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to fetch image from URL: {e}"
            )
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {e}"
            )
    
    @app.post("/predict/image", response_model=PredictionOutput)
    async def predict_image(file: UploadFile = File(...)):
        """
        Run inference on an uploaded image.
        
        Args:
            file: Uploaded image file
            
        Returns:
            Prediction results
        """
        try:
            # Read image file
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            # Run inference
            return await _predict_image(image)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {e}"
            )
    
    return app


def main():
    """Run the inference API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CLIP HAR Inference API")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model file")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["pytorch", "onnx", "torchscript", "tensorrt"],
        required=True,
        help="Type of the model"
    )
    parser.add_argument("--class_names", type=str, nargs="+", help="List of class names")
    parser.add_argument("--class_names_path", type=str, help="Path to a JSON file with class names")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top predictions to return")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    
    args = parser.parse_args()
    
    # Create FastAPI app
    app = create_app(
        model_path=args.model_path,
        model_type=args.model_type,
        class_names=args.class_names,
        class_names_path=args.class_names_path,
        top_k=args.top_k,
        device=args.device
    )
    
    # Run the server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
