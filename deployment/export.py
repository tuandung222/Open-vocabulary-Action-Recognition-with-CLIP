import os
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


def export_to_onnx(
    model: torch.nn.Module,
    save_path: str,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    opset_version: int = 13,
    simplify: bool = True,
    verbose: bool = True,
) -> str:
    """
    Export PyTorch CLIP model to ONNX format.

    Args:
        model: The PyTorch CLIP model to export
        save_path: Path where the ONNX model will be saved
        input_shape: Shape of the input tensor (batch_size, channels, height, width)
        dynamic_axes: Dynamic axes for variable length inputs/outputs
        opset_version: ONNX opset version to use
        simplify: Whether to simplify the ONNX model (requires onnx-simplifier)
        verbose: Whether to print export details

    Returns:
        Path to the exported ONNX model
    """
    if verbose:
        print(f"Exporting model to ONNX format at {save_path}")

    # Ensure the save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create dummy input
    dummy_input = torch.randn(input_shape)

    # Set model to evaluation mode
    model.eval()

    # Set default dynamic_axes if not provided
    if dynamic_axes is None:
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        verbose=verbose,
    )

    # Optionally simplify the model
    if simplify:
        try:
            import onnx
            from onnxsim import simplify

            # Load the ONNX model
            onnx_model = onnx.load(save_path)

            # Simplify
            simplified_model, check = simplify(onnx_model)
            assert check, "Simplified ONNX model could not be validated"

            # Save the simplified model
            onnx.save(simplified_model, save_path)
            if verbose:
                print(f"ONNX model simplified successfully")
        except ImportError:
            print("Warning: onnx-simplifier not installed. Skipping simplification.")
            print("Install with: pip install onnx-simplifier")

    if verbose:
        print(f"ONNX export complete: {save_path}")

    return save_path


def export_to_tensorrt(
    onnx_path: str,
    save_path: str,
    precision: str = "fp16",
    max_workspace_size: int = 1 << 30,  # 1GB
    max_batch_size: int = 16,
    verbose: bool = True,
) -> str:
    """
    Convert ONNX model to TensorRT for faster inference on NVIDIA GPUs.

    Args:
        onnx_path: Path to the ONNX model
        save_path: Path where the TensorRT engine will be saved
        precision: Precision to use ('fp32', 'fp16', or 'int8')
        max_workspace_size: Maximum workspace size in bytes
        max_batch_size: Maximum batch size for the TensorRT engine
        verbose: Whether to print conversion details

    Returns:
        Path to the TensorRT engine
    """
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
    except ImportError:
        raise ImportError(
            "TensorRT and/or PyCUDA not found. Install with: "
            "pip install nvidia-tensorrt pycuda"
        )

    if verbose:
        print(f"Converting ONNX model to TensorRT: {onnx_path}")

    # Ensure the save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create TensorRT logger
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)

    # Create builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size

    # Set precision flags
    if precision == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        if verbose:
            print("Using FP16 precision")
    elif precision == "int8" and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        # Note: INT8 requires calibration, which is not implemented here
        if verbose:
            print("Using INT8 precision")
    else:
        if verbose:
            print("Using FP32 precision")

    # Parse ONNX file
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(f"TensorRT ONNX parser error: {parser.get_error(error)}")
            raise RuntimeError("Failed to parse ONNX model")

    # Build engine
    if verbose:
        print("Building TensorRT engine (this may take a while)...")

    profile = builder.create_optimization_profile()
    input_name = "input"  # Must match the input name in ONNX export
    input_shape = network.get_input(0).shape

    # Set dynamic batch size
    min_batch, opt_batch = 1, max(1, max_batch_size // 2)
    profile.set_shape(
        input_name,
        (min_batch, input_shape[1], input_shape[2], input_shape[3]),
        (opt_batch, input_shape[1], input_shape[2], input_shape[3]),
        (max_batch_size, input_shape[1], input_shape[2], input_shape[3]),
    )
    config.add_optimization_profile(profile)

    engine = builder.build_engine(network, config)

    # Serialize engine and save to disk
    with open(save_path, "wb") as f:
        f.write(engine.serialize())

    if verbose:
        print(f"TensorRT engine saved to: {save_path}")

    return save_path


def validate_exported_model(
    pytorch_model: torch.nn.Module,
    onnx_path: str,
    input_data: Optional[torch.Tensor] = None,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    rtol: float = 1e-3,
    atol: float = 1e-5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[bool, Dict]:
    """
    Validate that the exported ONNX model produces the same outputs as the PyTorch model.

    Args:
        pytorch_model: Original PyTorch model
        onnx_path: Path to the exported ONNX model
        input_data: Optional input tensor for validation
        input_shape: Shape of input tensor if input_data is not provided
        rtol: Relative tolerance for output comparison
        atol: Absolute tolerance for output comparison
        device: Device to run the PyTorch model on

    Returns:
        Tuple of (is_valid, results_dict)
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "onnxruntime not found. Install with: pip install onnxruntime-gpu"
        )

    # Set model to evaluation mode
    pytorch_model.eval()
    pytorch_model.to(device)

    # Create input data if not provided
    if input_data is None:
        input_data = torch.randn(input_shape)

    # Get PyTorch model prediction
    with torch.no_grad():
        input_tensor = input_data.to(device)
        pytorch_output = pytorch_model(input_tensor).cpu().numpy()

    # Get ONNX model prediction
    ort_session = ort.InferenceSession(
        onnx_path,
        providers=["CUDAExecutionProvider"]
        if device == "cuda"
        else ["CPUExecutionProvider"],
    )
    ort_inputs = {ort_session.get_inputs()[0].name: input_data.numpy()}
    onnx_output = ort_session.run(None, ort_inputs)[0]

    # Compare outputs
    is_close = np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol)
    max_diff = np.max(np.abs(pytorch_output - onnx_output))
    mean_diff = np.mean(np.abs(pytorch_output - onnx_output))

    results = {
        "is_valid": is_close,
        "max_absolute_diff": float(max_diff),
        "mean_absolute_diff": float(mean_diff),
        "pytorch_output_shape": pytorch_output.shape,
        "onnx_output_shape": onnx_output.shape,
    }

    return is_close, results


def benchmark_model(
    model_path: str,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    model_type: str = "onnx",
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict:
    """
    Benchmark inference speed of the exported model.

    Args:
        model_path: Path to the model
        input_shape: Shape of the input tensor
        model_type: Type of model ('pytorch', 'onnx', 'tensorrt')
        num_iterations: Number of inference iterations for benchmarking
        warmup_iterations: Number of warmup iterations
        device: Device to run the benchmark on

    Returns:
        Dictionary with benchmark results
    """
    results = {
        "model_type": model_type,
        "device": device,
        "input_shape": input_shape,
        "iterations": num_iterations,
    }

    # Create dummy input
    dummy_input = torch.randn(input_shape)

    if model_type == "pytorch":
        # Load PyTorch model
        model = torch.load(model_path)
        model.eval()
        model.to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(dummy_input.to(device))

        # Time inference
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input.to(device))

        # Synchronize GPU
        if device == "cuda":
            torch.cuda.synchronize()

    elif model_type == "onnx":
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime not found. Install with: pip install onnxruntime-gpu"
            )

        # Create session
        providers = (
            ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        )
        session = ort.InferenceSession(model_path, providers=providers)
        input_name = session.get_inputs()[0].name

        # Warmup
        for _ in range(warmup_iterations):
            _ = session.run(None, {input_name: dummy_input.numpy()})

        # Time inference
        start_time = time.time()
        for _ in range(num_iterations):
            _ = session.run(None, {input_name: dummy_input.numpy()})

    elif model_type == "tensorrt":
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            raise ImportError("TensorRT and/or PyCUDA not found")

        # Load TensorRT engine
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        # Create execution context
        context = engine.create_execution_context()

        # Allocate memory
        h_input = cuda.pagelocked_empty(dummy_input.numpy().shape, dtype=np.float32)
        h_output = cuda.pagelocked_empty(
            (engine.get_binding_shape(1)[0], engine.get_binding_shape(1)[1]),
            dtype=np.float32,
        )
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)
        stream = cuda.Stream()

        # Warmup
        for _ in range(warmup_iterations):
            np.copyto(h_input, dummy_input.numpy())
            cuda.memcpy_htod_async(d_input, h_input, stream)
            context.execute_async_v2(
                bindings=[int(d_input), int(d_output)], stream_handle=stream.handle
            )
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            stream.synchronize()

        # Time inference
        start_time = time.time()
        for _ in range(num_iterations):
            np.copyto(h_input, dummy_input.numpy())
            cuda.memcpy_htod_async(d_input, h_input, stream)
            context.execute_async_v2(
                bindings=[int(d_input), int(d_output)], stream_handle=stream.handle
            )
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            stream.synchronize()

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Calculate results
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    fps = num_iterations / total_time

    results.update(
        {
            "total_time_seconds": total_time,
            "average_inference_ms": avg_time * 1000,
            "fps": fps,
        }
    )

    return results
