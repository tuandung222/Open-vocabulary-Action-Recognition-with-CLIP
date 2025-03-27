"""
Model optimization utilities for the CLIP HAR project.

This module provides functions for optimizing models using TensorRT,
including conversion, benchmarking, and calibration.
"""

import os
import time
import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union

# Set up logging
logger = logging.getLogger(__name__)

# Import TensorRT conditionally to avoid errors if not installed
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    logger.warning(
        "TensorRT and/or PyCUDA not available. TensorRT optimization will not work."
    )
    TENSORRT_AVAILABLE = False


class TensorRTOptimizer:
    """Optimizer for TensorRT model conversion and inference."""
    
    def __init__(
        self,
        precision: str = "fp16",
        max_workspace_size: int = 1 << 30,  # 1GB
        max_batch_size: int = 32,
        calibrator: Optional[object] = None,
    ):
        """
        Initialize the TensorRT optimizer.
        
        Args:
            precision: Precision to use for TensorRT optimization. Options: 'fp32', 'fp16', 'int8'
            max_workspace_size: Maximum workspace size in bytes
            max_batch_size: Maximum batch size for the engine
            calibrator: Optional calibrator for INT8 quantization
        """
        if not TENSORRT_AVAILABLE:
            raise ImportError(
                "TensorRT and/or PyCUDA are not available. "
                "Please install them to use TensorRTOptimizer."
            )
            
        self.precision = precision
        self.max_workspace_size = max_workspace_size
        self.max_batch_size = max_batch_size
        self.calibrator = calibrator
        
        # Initialize TensorRT
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.builder = trt.Builder(self.logger)
        self.network = None
        self.config = None
        self.parser = None
        self.engine = None
        self.context = None
        self.bindings = None
        self.stream = None
        
    def build_engine_from_onnx(self, onnx_path: str) -> trt.ICudaEngine:
        """
        Build a TensorRT engine from an ONNX model.
        
        Args:
            onnx_path: Path to the ONNX model file
            
        Returns:
            TensorRT engine
        """
        logger.info(f"Building TensorRT engine from ONNX model: {onnx_path}")
        
        # Create network definition
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.network = self.builder.create_network(network_flags)
        
        # Create parser and parse ONNX model
        self.parser = trt.OnnxParser(self.network, self.logger)
        
        # Parse ONNX file
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                for error in range(self.parser.num_errors):
                    logger.error(f"TensorRT ONNX parser error: {self.parser.get_error(error)}")
                raise RuntimeError(f"Failed to parse ONNX file: {onnx_path}")
        
        # Create builder configuration
        self.config = self.builder.create_builder_config()
        self.config.max_workspace_size = self.max_workspace_size
        
        # Set precision flags
        if self.precision == "fp16" and self.builder.platform_has_fast_fp16:
            logger.info("Enabling FP16 precision")
            self.config.set_flag(trt.BuilderFlag.FP16)
        elif self.precision == "int8" and self.builder.platform_has_fast_int8:
            logger.info("Enabling INT8 precision")
            self.config.set_flag(trt.BuilderFlag.INT8)
            if self.calibrator:
                self.config.int8_calibrator = self.calibrator
            else:
                logger.warning("INT8 precision requires a calibrator, but none was provided")
        
        # Set optimization profiles for dynamic shapes if needed
        profile = self.builder.create_optimization_profile()
        
        # Analyze network inputs and create optimization profiles
        for i in range(self.network.num_inputs):
            input_tensor = self.network.get_input(i)
            name = input_tensor.name
            shape = input_tensor.shape
            
            # Handle dynamic batch dimension
            if shape[0] == -1:
                min_shape = (1,) + tuple(shape[1:])
                opt_shape = (self.max_batch_size // 2,) + tuple(shape[1:])
                max_shape = (self.max_batch_size,) + tuple(shape[1:])
                
                profile.set_shape(name, min_shape, opt_shape, max_shape)
                logger.info(f"Setting dynamic shape for input {name}: {min_shape} -> {max_shape}")
        
        self.config.add_optimization_profile(profile)
        
        # Build the engine
        logger.info("Building TensorRT engine (this may take a while)...")
        plan = self.builder.build_serialized_network(self.network, self.config)
        engine = trt.Runtime(self.logger).deserialize_cuda_engine(plan)
        
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        logger.info("TensorRT engine built successfully")
        self.engine = engine
        return engine
    
    def save_engine(self, engine_path: str) -> None:
        """
        Save the TensorRT engine to a file.
        
        Args:
            engine_path: Path to save the engine
        """
        if self.engine is None:
            raise RuntimeError("No engine to save. Build an engine first.")
        
        logger.info(f"Saving TensorRT engine to: {engine_path}")
        with open(engine_path, "wb") as f:
            f.write(self.engine.serialize())
    
    def load_engine(self, engine_path: str) -> trt.ICudaEngine:
        """
        Load a TensorRT engine from a file.
        
        Args:
            engine_path: Path to the TensorRT engine file
            
        Returns:
            TensorRT engine
        """
        logger.info(f"Loading TensorRT engine from: {engine_path}")
        with open(engine_path, "rb") as f:
            plan = f.read()
        
        runtime = trt.Runtime(self.logger)
        engine = runtime.deserialize_cuda_engine(plan)
        
        if engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from: {engine_path}")
        
        self.engine = engine
        return engine
    
    def prepare_inference(self) -> None:
        """
        Prepare the engine for inference by creating execution context,
        allocating device memory, and creating a CUDA stream.
        """
        if self.engine is None:
            raise RuntimeError("No engine available. Build or load an engine first.")
        
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        # Allocate device memory
        self.bindings = []
        self.input_shapes = {}
        
        for binding_idx in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(binding_idx)
            shape = self.engine.get_binding_shape(binding_idx)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding_idx))
            
            # Allocate device memory
            if self.engine.binding_is_input(binding_idx):
                self.input_shapes[name] = shape
                # If first dim is -1, it's a dynamic shape
                if shape[0] == -1:
                    shape = (self.max_batch_size,) + tuple(shape[1:])
                    self.context.set_binding_shape(binding_idx, shape)
            
            size = trt.volume(shape) * dtype.itemsize
            device_memory = cuda.mem_alloc(size)
            self.bindings.append(int(device_memory))
        
        # Create CUDA stream
        self.stream = cuda.Stream()
    
    def infer(
        self, inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Run inference with the TensorRT engine.
        
        Args:
            inputs: Dictionary of input names to numpy arrays
            
        Returns:
            Dictionary of output names to numpy arrays
        """
        if self.context is None:
            self.prepare_inference()
        
        # Get input and output binding indices
        input_binding_idxs = []
        output_binding_idxs = []
        
        for binding_idx in range(self.engine.num_bindings):
            if self.engine.binding_is_input(binding_idx):
                input_binding_idxs.append(binding_idx)
            else:
                output_binding_idxs.append(binding_idx)
        
        # Set input shapes and copy input data to GPU
        host_inputs = []
        for binding_idx in input_binding_idxs:
            name = self.engine.get_binding_name(binding_idx)
            
            if name not in inputs:
                raise ValueError(f"Input '{name}' not provided")
            
            input_data = inputs[name].astype(trt.nptype(self.engine.get_binding_dtype(binding_idx)))
            host_inputs.append(input_data)
            
            # Set dynamic shape if necessary
            if self.engine.get_binding_shape(binding_idx)[0] == -1:
                shape = (input_data.shape[0],) + tuple(self.engine.get_binding_shape(binding_idx)[1:])
                self.context.set_binding_shape(binding_idx, shape)
            
            # Copy input data to device
            cuda.memcpy_htod_async(
                self.bindings[binding_idx], 
                input_data.ravel(),
                self.stream
            )
        
        # Prepare output arrays
        outputs = {}
        host_outputs = []
        
        for binding_idx in output_binding_idxs:
            name = self.engine.get_binding_name(binding_idx)
            shape = self.context.get_binding_shape(binding_idx)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding_idx))
            
            # Allocate host memory for output
            host_output = np.empty(shape, dtype=dtype)
            host_outputs.append(host_output)
            
            outputs[name] = host_output
        
        # Run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        
        # Copy outputs from device to host
        for binding_idx, host_output in zip(output_binding_idxs, host_outputs):
            cuda.memcpy_dtoh_async(
                host_output, 
                self.bindings[binding_idx],
                self.stream
            )
        
        # Synchronize the stream
        self.stream.synchronize()
        
        return outputs
    
    def benchmark(
        self,
        inputs: Dict[str, np.ndarray],
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark the TensorRT engine.
        
        Args:
            inputs: Dictionary of input names to numpy arrays
            num_iterations: Number of iterations to run
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Dictionary with benchmark results (latency_ms, throughput_fps)
        """
        if self.context is None:
            self.prepare_inference()
        
        # Warmup
        logger.info(f"Running {warmup_iterations} warmup iterations")
        for _ in range(warmup_iterations):
            self.infer(inputs)
        
        # Benchmark
        logger.info(f"Running {num_iterations} benchmark iterations")
        start_time = time.time()
        
        for _ in range(num_iterations):
            self.infer(inputs)
            
        end_time = time.time()
        
        # Calculate benchmark results
        elapsed_time = end_time - start_time
        latency_ms = (elapsed_time / num_iterations) * 1000
        throughput_fps = num_iterations / elapsed_time
        
        results = {
            "latency_ms": latency_ms,
            "throughput_fps": throughput_fps,
            "precision": self.precision,
            "batch_size": inputs[list(inputs.keys())[0]].shape[0]
        }
        
        logger.info(f"Benchmark results: {results}")
        return results
    
    def cleanup(self) -> None:
        """Release all resources."""
        self.context = None
        self.engine = None
        self.network = None
        self.config = None
        self.parser = None
        self.bindings = None
        self.stream = None


def optimize_with_tensorrt(
    onnx_model_path: str,
    output_engine_path: str,
    precision: str = "fp16",
    max_batch_size: int = 32,
    run_benchmark: bool = True,
    input_shapes: Optional[Dict[str, Tuple[int, ...]]] = None
) -> Dict[str, float]:
    """
    Convert an ONNX model to TensorRT and optionally benchmark it.
    
    Args:
        onnx_model_path: Path to the ONNX model file
        output_engine_path: Path to save the TensorRT engine
        precision: Precision to use for TensorRT optimization ('fp32', 'fp16', 'int8')
        max_batch_size: Maximum batch size for the engine
        run_benchmark: Whether to run a benchmark after conversion
        input_shapes: Optional dictionary of input names to shapes for benchmarking
        
    Returns:
        Dictionary with benchmark results if run_benchmark is True
    """
    if not TENSORRT_AVAILABLE:
        raise ImportError(
            "TensorRT and/or PyCUDA are not available. "
            "Please install them to use optimize_with_tensorrt."
        )
    
    # Create TensorRT optimizer
    optimizer = TensorRTOptimizer(
        precision=precision,
        max_batch_size=max_batch_size
    )
    
    # Build engine from ONNX model
    optimizer.build_engine_from_onnx(onnx_model_path)
    
    # Save engine
    optimizer.save_engine(output_engine_path)
    
    results = {}
    
    # Run benchmark if requested
    if run_benchmark and input_shapes:
        # Create random input data for benchmarking
        inputs = {}
        for name, shape in input_shapes.items():
            inputs[name] = np.random.rand(*shape).astype(np.float32)
        
        # Run benchmark
        results = optimizer.benchmark(inputs)
        
        logger.info(f"Model: {os.path.basename(onnx_model_path)}")
        logger.info(f"Precision: {precision}")
        logger.info(f"Batch size: {max_batch_size}")
        logger.info(f"Latency: {results['latency_ms']:.2f} ms")
        logger.info(f"Throughput: {results['throughput_fps']:.2f} FPS")
    
    # Cleanup
    optimizer.cleanup()
    
    return results 