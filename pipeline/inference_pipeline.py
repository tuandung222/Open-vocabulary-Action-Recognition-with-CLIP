#!/usr/bin/env python
# Inference pipeline for HAR classification using CLIP

import os
import sys
import argparse
import torch
import logging
import json
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union

# Import project modules
from CLIP_HAR_PROJECT.configs.default import get_config
from CLIP_HAR_PROJECT.models.clip_model import CLIPLabelRetriever
from CLIP_HAR_PROJECT.data.preprocessing import transform_image
from CLIP_HAR_PROJECT.mlops.tracking import load_model_from_mlflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("inference_pipeline")


class ModelLoader:
    """
    Utility class for loading models from different sources and formats.
    """
    
    @staticmethod
    def load_pytorch_model(
        model_path: str,
        model_name: Optional[str] = "openai/clip-vit-base-patch16",
        labels: Optional[List[str]] = None,
        prompt_template: Optional[str] = "a photo of person/people who is/are {label}",
        device: Optional[str] = None
    ) -> CLIPLabelRetriever:
        """
        Load a PyTorch CLIP model.
        
        Args:
            model_path: Path to the model checkpoint
            model_name: Name of the base CLIP model
            labels: List of class labels
            prompt_template: Template for text prompts
            device: Device to load the model on
        
        Returns:
            Loaded model
        """
        logger.info(f"Loading PyTorch model from {model_path}")
        
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract model configuration
        if "labels" in checkpoint:
            labels = checkpoint["labels"]
        elif "class_names" in checkpoint:
            labels = checkpoint["class_names"]
            
        if "prompt_template" in checkpoint:
            prompt_template = checkpoint["prompt_template"]
        
        # Ensure we have labels
        if labels is None:
            raise ValueError("Labels not found in checkpoint and not provided.")
        
        # Create model
        model = CLIPLabelRetriever.from_pretrained(
            model_name,
            labels=labels,
            prompt_template=prompt_template
        )
        
        # Load state dict
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)
        
        # Move model to device
        model = model.to(device)
        model.eval()
        
        return model
    
    @staticmethod
    def load_mlflow_model(
        model_uri: str,
        device: Optional[str] = None
    ) -> CLIPLabelRetriever:
        """
        Load a model from MLflow.
        
        Args:
            model_uri: URI of the model in MLflow
            device: Device to load the model on
        
        Returns:
            Loaded model
        """
        logger.info(f"Loading model from MLflow: {model_uri}")
        
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model from MLflow
        model = load_model_from_mlflow(model_uri)
        
        # Move model to device
        model = model.to(device)
        model.eval()
        
        return model
    
    @staticmethod
    def load_onnx_model(
        model_path: str
    ) -> Any:
        """
        Load an ONNX model.
        
        Args:
            model_path: Path to the ONNX model
        
        Returns:
            ONNX inference session
        """
        logger.info(f"Loading ONNX model from {model_path}")
        
        try:
            import onnxruntime as ort
            
            # Create ONNX Runtime session with appropriate provider
            providers = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            session = ort.InferenceSession(model_path, providers=providers)
            
            return session
        except ImportError:
            raise ImportError("ONNX Runtime not installed. Install with: pip install onnxruntime-gpu or onnxruntime")
    
    @staticmethod
    def load_tensorrt_model(
        model_path: str
    ) -> Tuple[Any, Dict]:
        """
        Load a TensorRT engine.
        
        Args:
            model_path: Path to the TensorRT engine
        
        Returns:
            Tuple of (TensorRT execution context, binding information)
        """
        logger.info(f"Loading TensorRT engine from {model_path}")
        
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            # Create TensorRT logger
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            # Load TensorRT engine
            with open(model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
            
            # Create execution context
            context = engine.create_execution_context()
            
            # Prepare binding information
            binding_info = {
                "input_binding_idx": engine.get_binding_index("input"),
                "output_binding_idx": engine.get_binding_index("output"),
                "input_shape": engine.get_binding_shape(0),
                "output_shape": engine.get_binding_shape(1),
                "input_dtype": trt.nptype(engine.get_binding_dtype(0)),
                "output_dtype": trt.nptype(engine.get_binding_dtype(1))
            }
            
            return context, binding_info
        except ImportError:
            raise ImportError("TensorRT or PyCUDA not installed. Install with: pip install tensorrt pycuda")


class InferencePipeline:
    """
    End-to-end inference pipeline for HAR classification using CLIP.
    
    This pipeline handles:
    1. Model loading (PyTorch, ONNX, TensorRT)
    2. Input preprocessing
    3. Model inference
    4. Output formatting
    
    Supports:
    - Single image inference
    - Batch inference
    - Video inference
    - Multiple model formats
    """
    
    def __init__(
        self,
        model_path: str,
        model_type: str = "pytorch",
        model_name: Optional[str] = "openai/clip-vit-base-patch16",
        labels: Optional[List[str]] = None,
        device: Optional[str] = None,
        image_size: int = 224,
        class_confidence_threshold: float = 0.5
    ):
        """
        Initialize the inference pipeline.
        
        Args:
            model_path: Path to the model file
            model_type: Type of model (pytorch, mlflow, onnx, tensorrt)
            model_name: Name of the base CLIP model (for PyTorch model)
            labels: List of class labels (for ONNX/TensorRT models)
            device: Device to run inference on
            image_size: Size of the input image
            class_confidence_threshold: Threshold for class confidence
        """
        # Store parameters
        self.model_path = model_path
        self.model_type = model_type
        self.model_name = model_name
        self.labels = labels
        self.image_size = image_size
        self.class_confidence_threshold = class_confidence_threshold
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize model-specific attributes
        self.model = None
        self.onnx_session = None
        self.tensorrt_context = None
        self.tensorrt_binding_info = None
        self.cuda_stream = None
        
        # Load model
        self._load_model()
        
        # If labels were not provided but obtained from the model, store them
        if self.labels is None and hasattr(self.model, "labels"):
            self.labels = self.model.labels
        
        # Validate that we have labels
        if self.labels is None and model_type in ["onnx", "tensorrt"]:
            raise ValueError("Labels must be provided for ONNX and TensorRT models.")
        
        logger.info(f"Inference pipeline initialized with model type: {model_type}")
        
        # Initialize TensorRT specific resources if needed
        if model_type == "tensorrt":
            import pycuda.driver as cuda
            self.cuda_stream = cuda.Stream()
    
    def _load_model(self):
        """Load the model based on the specified type."""
        if self.model_type == "pytorch":
            self.model = ModelLoader.load_pytorch_model(
                model_path=self.model_path,
                model_name=self.model_name,
                labels=self.labels,
                device=self.device
            )
            # Get labels from model if not provided
            if self.labels is None:
                self.labels = self.model.labels
                
        elif self.model_type == "mlflow":
            self.model = ModelLoader.load_mlflow_model(
                model_uri=self.model_path,
                device=self.device
            )
            # Get labels from model
            self.labels = self.model.labels
            
        elif self.model_type == "onnx":
            self.onnx_session = ModelLoader.load_onnx_model(
                model_path=self.model_path
            )
            
        elif self.model_type == "tensorrt":
            self.tensorrt_context, self.tensorrt_binding_info = ModelLoader.load_tensorrt_model(
                model_path=self.model_path
            )
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _preprocess_image(
        self,
        image: Union[str, np.ndarray, Image.Image]
    ) -> torch.Tensor:
        """
        Preprocess an image for inference.
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
        
        Returns:
            Preprocessed image tensor
        """
        # Load image if path is provided
        if isinstance(image, str):
            # Check if file exists
            if not os.path.exists(image):
                raise ValueError(f"Image file not found: {image}")
            
            # Load image
            try:
                image = Image.open(image).convert("RGB")
            except Exception as e:
                raise ValueError(f"Error loading image: {e}")
        
        # Convert numpy array to PIL Image
        elif isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Ensure image is a PIL Image
        if not isinstance(image, Image.Image):
            raise ValueError("Image must be a file path, numpy array, or PIL Image")
        
        # Preprocess image based on model type
        if self.model_type in ["pytorch", "mlflow"]:
            # Use the model's preprocessing function if available
            if hasattr(self.model, "preprocess_image"):
                return self.model.preprocess_image(image).to(self.device)
            else:
                # Generic preprocessing
                return transform_image(image, self.image_size).to(self.device)
        
        elif self.model_type == "onnx":
            # ONNX models expect numpy input
            transformed = transform_image(image, self.image_size).numpy()
            return transformed
        
        elif self.model_type == "tensorrt":
            # TensorRT models expect numpy input with appropriate data type
            transformed = transform_image(image, self.image_size).numpy()
            return transformed.astype(self.tensorrt_binding_info["input_dtype"])
    
    def _pytorch_inference(
        self,
        processed_image: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Run inference using PyTorch model.
        
        Args:
            processed_image: Preprocessed image tensor
        
        Returns:
            Dictionary with prediction results
        """
        with torch.no_grad():
            # Get predictions
            text_embeds, visual_embeds, scores = self.model.predict(processed_image)
            
            # Get predicted class and confidence
            class_idx = scores.argmax(axis=1)[0]
            confidence = scores[0, class_idx]
            predicted_class = self.labels[class_idx]
            
            # Get top-k predictions
            top_k = min(3, len(self.labels))
            top_indices = np.argsort(scores[0])[::-1][:top_k]
            top_classes = [self.labels[idx] for idx in top_indices]
            top_scores = scores[0, top_indices]
            
            # Create result dictionary
            result = {
                "predicted_class": predicted_class,
                "confidence": float(confidence),
                "class_idx": int(class_idx),
                "top_classes": top_classes,
                "top_scores": top_scores.tolist(),
                "all_scores": scores[0].tolist(),
                "text_embeds": text_embeds.tolist() if text_embeds is not None else None,
                "visual_embeds": visual_embeds.tolist() if visual_embeds is not None else None
            }
            
            return result
    
    def _onnx_inference(
        self,
        processed_image: np.ndarray
    ) -> Dict[str, Any]:
        """
        Run inference using ONNX model.
        
        Args:
            processed_image: Preprocessed image as numpy array
        
        Returns:
            Dictionary with prediction results
        """
        # Get input name
        input_name = self.onnx_session.get_inputs()[0].name
        
        # Run inference
        outputs = self.onnx_session.run(None, {input_name: processed_image})
        scores = outputs[0][0]  # Assuming output[0] is the scores
        
        # Get predicted class and confidence
        class_idx = np.argmax(scores)
        confidence = scores[class_idx]
        predicted_class = self.labels[class_idx]
        
        # Get top-k predictions
        top_k = min(3, len(self.labels))
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_classes = [self.labels[idx] for idx in top_indices]
        top_scores = scores[top_indices]
        
        # Create result dictionary
        result = {
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "class_idx": int(class_idx),
            "top_classes": top_classes,
            "top_scores": top_scores.tolist(),
            "all_scores": scores.tolist(),
            "text_embeds": None,  # Not available in ONNX models
            "visual_embeds": None  # Not available in ONNX models
        }
        
        return result
    
    def _tensorrt_inference(
        self,
        processed_image: np.ndarray
    ) -> Dict[str, Any]:
        """
        Run inference using TensorRT engine.
        
        Args:
            processed_image: Preprocessed image as numpy array
        
        Returns:
            Dictionary with prediction results
        """
        import pycuda.driver as cuda
        
        # Get binding information
        input_binding_idx = self.tensorrt_binding_info["input_binding_idx"]
        output_binding_idx = self.tensorrt_binding_info["output_binding_idx"]
        output_shape = self.tensorrt_binding_info["output_shape"]
        output_dtype = self.tensorrt_binding_info["output_dtype"]
        
        # Allocate memory
        h_input = processed_image.astype(self.tensorrt_binding_info["input_dtype"])
        h_output = np.empty(output_shape, dtype=output_dtype)
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)
        
        # Copy input to device
        cuda.memcpy_htod_async(d_input, h_input, self.cuda_stream)
        
        # Run inference
        bindings = [int(d_input), int(d_output)]
        self.tensorrt_context.execute_async_v2(
            bindings=bindings,
            stream_handle=self.cuda_stream.handle
        )
        
        # Copy output to host
        cuda.memcpy_dtoh_async(h_output, d_output, self.cuda_stream)
        
        # Synchronize stream
        self.cuda_stream.synchronize()
        
        # Process results
        scores = h_output[0]  # Assuming output[0] is the scores
        
        # Get predicted class and confidence
        class_idx = np.argmax(scores)
        confidence = scores[class_idx]
        predicted_class = self.labels[class_idx]
        
        # Get top-k predictions
        top_k = min(3, len(self.labels))
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_classes = [self.labels[idx] for idx in top_indices]
        top_scores = scores[top_indices]
        
        # Create result dictionary
        result = {
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "class_idx": int(class_idx),
            "top_classes": top_classes,
            "top_scores": top_scores.tolist(),
            "all_scores": scores.tolist(),
            "text_embeds": None,  # Not available in TensorRT models
            "visual_embeds": None  # Not available in TensorRT models
        }
        
        return result
    
    def predict(
        self,
        image: Union[str, np.ndarray, Image.Image]
    ) -> Dict[str, Any]:
        """
        Run inference on a single image.
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
        
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        processed_image = self._preprocess_image(image)
        
        # Run model-specific inference
        if self.model_type in ["pytorch", "mlflow"]:
            processed_image = processed_image.unsqueeze(0)  # Add batch dimension
            result = self._pytorch_inference(processed_image)
        elif self.model_type == "onnx":
            result = self._onnx_inference(processed_image)
        elif self.model_type == "tensorrt":
            result = self._tensorrt_inference(processed_image)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return result
    
    def batch_predict(
        self,
        images: List[Union[str, np.ndarray, Image.Image]]
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of images.
        
        Args:
            images: List of input images
        
        Returns:
            List of dictionaries with prediction results
        """
        results = []
        
        # Process each image individually
        for image in images:
            try:
                result = self.predict(image)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                results.append({"error": str(e)})
        
        return results
    
    def predict_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        fps: Optional[int] = None,
        frame_interval: int = 1,
        show_display: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Run inference on video frames.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            fps: Frames per second for output video (optional)
            frame_interval: Process every N frames
            show_display: Whether to display video during processing
        
        Returns:
            List of prediction results for each processed frame
        """
        # Ensure video file exists
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Set output fps
        if fps is None:
            fps = original_fps
        
        # Create video writer if output path is provided
        writer = None
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_idx = 0
        results = []
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Total frames: {frame_count}, FPS: {original_fps}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every N frames
            if frame_idx % frame_interval == 0:
                # Run inference
                try:
                    result = self.predict(frame)
                    results.append(result)
                    
                    # Add prediction overlay
                    label = result["predicted_class"]
                    conf = result["confidence"]
                    cv2.putText(
                        frame, 
                        f"{label}: {conf:.2f}", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0), 
                        2
                    )
                    
                    # Log progress
                    if frame_idx % (frame_interval * 10) == 0:
                        logger.info(f"Processed frame {frame_idx}/{frame_count} ({frame_idx/frame_count*100:.1f}%)")
                        
                except Exception as e:
                    logger.error(f"Error processing frame {frame_idx}: {e}")
            
            # Write frame to output video
            if writer:
                writer.write(frame)
            
            # Display frame if requested
            if show_display:
                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_idx += 1
        
        # Release resources
        cap.release()
        if writer:
            writer.release()
        if show_display:
            cv2.destroyAllWindows()
        
        logger.info(f"Video processing complete: {frame_idx} frames processed")
        
        return results
    
    def process_directory(
        self,
        input_dir: str,
        output_path: Optional[str] = None,
        recursive: bool = False,
        image_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp']
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Input directory path
            output_path: Path to save results JSON file (optional)
            recursive: Whether to search subdirectories
            image_extensions: List of image file extensions to process
        
        Returns:
            Dictionary mapping file paths to prediction results
        """
        # Ensure directory exists
        if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
            raise ValueError(f"Directory not found: {input_dir}")
        
        # Find image files
        image_files = []
        if recursive:
            for root, _, files in os.walk(input_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_files.append(os.path.join(root, file))
        else:
            image_files = [
                os.path.join(input_dir, f) for f in os.listdir(input_dir)
                if os.path.isfile(os.path.join(input_dir, f)) and
                any(f.lower().endswith(ext) for ext in image_extensions)
            ]
        
        # Sort files for deterministic ordering
        image_files.sort()
        
        logger.info(f"Found {len(image_files)} images in {input_dir}")
        
        # Process each image
        results = {}
        for i, image_path in enumerate(image_files):
            try:
                # Extract relative path for cleaner output
                rel_path = os.path.relpath(image_path, input_dir)
                
                # Process image
                result = self.predict(image_path)
                results[rel_path] = result
                
                # Log progress
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i+1}/{len(image_files)} images ({(i+1)/len(image_files)*100:.1f}%)")
                    
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results[image_path] = {"error": str(e)}
        
        # Save results to file if requested
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_path}")
        
        return results
    
    def cleanup(self):
        """Release resources used by the pipeline."""
        # PyTorch/MLflow specific cleanup
        if self.model is not None and isinstance(self.model, torch.nn.Module):
            self.model.cpu()
            del self.model
        
        # TensorRT specific cleanup
        if self.cuda_stream is not None:
            self.cuda_stream.synchronize()
            del self.cuda_stream
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference pipeline for HAR classification")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model file")
    parser.add_argument("--model_type", type=str, default="pytorch",
                        choices=["pytorch", "mlflow", "onnx", "tensorrt"],
                        help="Type of model")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch16",
                        help="Name of the base CLIP model (for PyTorch models)")
    parser.add_argument("--labels_file", type=str, default=None,
                        help="Path to JSON file with class labels (for ONNX/TensorRT models)")
    
    # Input arguments (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str,
                           help="Path to input image")
    input_group.add_argument("--video", type=str,
                           help="Path to input video")
    input_group.add_argument("--directory", type=str,
                           help="Path to input directory")
    
    # Output arguments
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save output (depends on input type)")
    
    # Processing arguments
    parser.add_argument("--recursive", action="store_true",
                        help="Recursively search subdirectories (for directory input)")
    parser.add_argument("--frame_interval", type=int, default=1,
                        help="Process every N frames (for video input)")
    parser.add_argument("--show_display", action="store_true",
                        help="Show display during video processing")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Size of the input image")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                        help="Threshold for class confidence")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run inference on (cpu, cuda)")
    
    return parser.parse_args()


def load_labels_from_file(labels_file):
    """Load class labels from a JSON file."""
    with open(labels_file, 'r') as f:
        return json.load(f)


def main():
    """Main function to run the inference pipeline."""
    # Parse command line arguments
    args = parse_args()
    
    # Load labels from file if provided
    labels = None
    if args.labels_file:
        labels = load_labels_from_file(args.labels_file)
    
    try:
        # Create inference pipeline
        pipeline = InferencePipeline(
            model_path=args.model_path,
            model_type=args.model_type,
            model_name=args.model_name,
            labels=labels,
            device=args.device,
            image_size=args.image_size,
            class_confidence_threshold=args.confidence_threshold
        )
        
        # Run inference based on input type
        if args.image:
            # Process single image
            result = pipeline.predict(args.image)
            
            # Print result
            print(f"Prediction for {args.image}:")
            print(f"Class: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Top classes: {list(zip(result['top_classes'], [f'{s:.4f}' for s in result['top_scores']]))}")
            
            # Save result to file if requested
            if args.output:
                os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Result saved to {args.output}")
        
        elif args.video:
            # Process video
            results = pipeline.predict_video(
                video_path=args.video,
                output_path=args.output,
                frame_interval=args.frame_interval,
                show_display=args.show_display
            )
            
            # Print summary
            top_classes = {}
            for result in results:
                cls = result["predicted_class"]
                top_classes[cls] = top_classes.get(cls, 0) + 1
            
            top_classes = sorted(top_classes.items(), key=lambda x: x[1], reverse=True)
            print(f"Video analysis summary:")
            print(f"Total frames analyzed: {len(results)}")
            print(f"Top detected actions:")
            for cls, count in top_classes[:5]:
                print(f"  {cls}: {count} frames ({count/len(results)*100:.1f}%)")
        
        elif args.directory:
            # Process directory
            results = pipeline.process_directory(
                input_dir=args.directory,
                output_path=args.output,
                recursive=args.recursive
            )
            
            # Print summary
            top_classes = {}
            for result in results.values():
                if "predicted_class" in result:
                    cls = result["predicted_class"]
                    top_classes[cls] = top_classes.get(cls, 0) + 1
            
            top_classes = sorted(top_classes.items(), key=lambda x: x[1], reverse=True)
            print(f"Directory analysis summary:")
            print(f"Total images analyzed: {len(results)}")
            print(f"Top detected actions:")
            for cls, count in top_classes[:5]:
                print(f"  {cls}: {count} images ({count/len(results)*100:.1f}%)")
        
        return 0
    
    except Exception as e:
        logger.exception(f"Error in inference pipeline: {e}")
        return 1
    
    finally:
        # Clean up resources
        if 'pipeline' in locals():
            pipeline.cleanup()


if __name__ == "__main__":
    sys.exit(main()) 