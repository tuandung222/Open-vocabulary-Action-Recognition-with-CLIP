# Model Export Guide

## Why Export Models?

After training your deep learning model, you'll often want to **export** it to a format optimized for deployment. Exporting models to specialized formats offers several benefits:

1. **Faster Inference**: Optimized formats can run predictions much faster
2. **Smaller Size**: Exported models are often more compact
3. **Better Hardware Support**: Some formats are optimized for specific hardware
4. **Deployment Flexibility**: Different environments may require different formats
5. **Integration**: Makes it easier to use your model in various applications

In this guide, we'll explore how to export your CLIP HAR model to different formats, focusing on ONNX and TensorRT.

## Understanding Export Formats

### PyTorch Native Format (.pt/.pth)

This is the default format when you save a PyTorch model:
```python
torch.save(model.state_dict(), "model.pt")
```

While convenient for development, it's not optimized for deployment.

### ONNX (Open Neural Network Exchange)

ONNX is an open format designed to represent machine learning models in a standard way, enabling models to be transferred between different frameworks.

**Benefits**:
- Framework-agnostic (works with PyTorch, TensorFlow, etc.)
- Supported by many deployment platforms
- Enables optimizations and hardware acceleration
- Well-documented with strong community support

### TensorRT

NVIDIA TensorRT is a platform for high-performance deep learning inference, specifically optimized for NVIDIA GPUs.

**Benefits**:
- Maximum performance on NVIDIA hardware
- Reduced precision options (FP16, INT8) for faster inference
- Kernel fusion and other advanced optimizations
- Can provide 2-5x speedup over standard deployment

### TorchScript

TorchScript is a way to create serializable and optimizable models from PyTorch code:

**Benefits**:
- Stays within the PyTorch ecosystem
- Supports both tracing and scripting for conversion
- Good for deploying to C++ environments
- Maintains PyTorch's dynamic computation graph capabilities

## Export Process Overview

### Basic Flow

1. **Train your model** in PyTorch
2. **Prepare the model** for export (e.g., set to evaluation mode)
3. **Convert to target format** (ONNX, TensorRT, etc.)
4. **Validate the exported model** works correctly
5. **Optimize the exported model** for performance
6. **Deploy the optimized model**

## Exporting to ONNX

### Basic ONNX Export

Here's how to export a PyTorch model to ONNX:

```python
import torch
import torch.onnx

# Load your trained model
model = YourModel()
model.load_state_dict(torch.load('model.pt'))
model.eval()  # Set to evaluation mode

# Create dummy input (same shape as your model expects)
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model,               # model being run
    dummy_input,         # model input
    "model.onnx",        # output file
    export_params=True,  # store trained parameter weights inside the model
    opset_version=13,    # the ONNX version to export to
    do_constant_folding=True,  # optimization: fold constant ops
    input_names=['input'],     # input names
    output_names=['output'],   # output names
    dynamic_axes={             # variable length axes
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("Model exported to ONNX!")
```

### ONNX Optimization

Once exported, you can optimize your ONNX model:

```python
import onnx
from onnxsim import simplify

# Load the ONNX model
model = onnx.load("model.onnx")

# Verify the model
onnx.checker.check_model(model)

# Simplify the model
model_simplified, check = simplify(model)

if check:
    print("Simplified ONNX model validated!")
    onnx.save(model_simplified, "model_simplified.onnx")
else:
    print("Simplified ONNX model could not be validated!")
```

## Exporting to TensorRT

### Method 1: ONNX to TensorRT

The most common approach is to convert PyTorch → ONNX → TensorRT:

```python
import tensorrt as trt
import os

def build_engine_from_onnx(onnx_file_path):
    """Build TensorRT engine from ONNX model"""
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(EXPLICIT_BATCH) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:
        
        # Allow TensorRT to use up to 1GB of GPU memory
        builder.max_workspace_size = 1 << 30
        
        # Parse ONNX
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise RuntimeError(f"Failed to parse ONNX file: {onnx_file_path}")
        
        # Build engine
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        
        # Use FP16 precision if supported
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("Using FP16 mode")
        
        engine = builder.build_engine(network, config)
        
        # Save engine
        with open("model.trt", "wb") as f:
            f.write(engine.serialize())
        
        return engine

# Convert ONNX model to TensorRT
build_engine_from_onnx("model_simplified.onnx")
print("TensorRT engine created!")
```

### Method 2: Direct PyTorch to TensorRT (with Torch-TensorRT)

For a more integrated approach:

```python
import torch
import torch_tensorrt

# Load your PyTorch model
model = YourModel()
model.load_state_dict(torch.load('model.pt'))
model.eval()

# Compile with Torch-TensorRT
trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input(
        min_shape=[1, 3, 224, 224],
        opt_shape=[16, 3, 224, 224],
        max_shape=[32, 3, 224, 224],
        dtype=torch.float32
    )],
    enabled_precisions={torch.float16},  # Enable FP16
    workspace_size=1 << 30  # 1GB
)

# Save the compiled model
torch.jit.save(trt_model, "trt_model.ts")
print("Model exported with Torch-TensorRT!")
```

## CLIP HAR Export Module

In our project, we provide a specialized export module to make this process easy:

### Using Our Export Script

```bash
python -m CLIP_HAR_PROJECT.deployment.export_clip_model \
    --model_path outputs/trained_model.pt \
    --export_format onnx tensorrt \
    --benchmark
```

The script handles:
- Loading the trained CLIP HAR model
- Setting appropriate model configurations
- Handling model-specific requirements
- Exporting to the specified formats
- Running a benchmark to compare performance

### Export Configuration

Our export module supports various options:

```bash
python -m CLIP_HAR_PROJECT.deployment.export_clip_model --help
```

Key options include:
- `--export_format`: Specify one or more formats (onnx, tensorrt, torchscript)
- `--precision`: Precision to use (fp32, fp16, int8)
- `--optimize`: Apply format-specific optimizations
- `--dynamic_batch`: Support variable batch sizes
- `--benchmark`: Compare performance between formats

## Testing Exported Models

### Validating Output Consistency

It's crucial to check that your exported model produces the same outputs as the original:

```python
import torch
import onnxruntime
import numpy as np

# Prepare input
input_tensor = torch.randn(1, 3, 224, 224)
input_numpy = input_tensor.numpy()

# Get PyTorch output
with torch.no_grad():
    pytorch_output = original_model(input_tensor).numpy()

# Get ONNX output
ort_session = onnxruntime.InferenceSession("model.onnx")
onnx_output = ort_session.run(
    ['output'], 
    {'input': input_numpy}
)[0]

# Compare outputs
match = np.allclose(pytorch_output, onnx_output, rtol=1e-3, atol=1e-5)
print(f"Outputs match: {match}")

# Measure accuracy difference if needed
diff = np.abs(pytorch_output - onnx_output).mean()
print(f"Mean absolute difference: {diff}")
```

### Measuring Performance Gains

Benchmark your original vs. exported models:

```python
import time
import torch
import onnxruntime
import numpy as np

# Prepare batch of inputs
batch_size = 32
inputs = torch.randn(batch_size, 3, 224, 224)
input_numpy = inputs.numpy()

# Benchmark PyTorch
with torch.no_grad():
    start = time.time()
    for _ in range(100):  # Run 100 iterations
        _ = original_model(inputs)
    torch_time = (time.time() - start) / 100

# Benchmark ONNX
ort_session = onnxruntime.InferenceSession("model.onnx")
start = time.time()
for _ in range(100):  # Run 100 iterations
    _ = ort_session.run(['output'], {'input': input_numpy})
onnx_time = (time.time() - start) / 100

# Calculate speedup
speedup = torch_time / onnx_time
print(f"PyTorch inference time: {torch_time*1000:.2f} ms")
print(f"ONNX inference time: {onnx_time*1000:.2f} ms")
print(f"Speedup: {speedup:.2f}x")
```

## Common Export Issues and Solutions

### 1. Dynamic Shapes

**Problem**: Models with variable input sizes fail to export
**Solution**: Use dynamic axes in ONNX export
```python
dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
```

### 2. Custom Operations

**Problem**: Custom PyTorch operations aren't supported in ONNX
**Solution**: Register custom ONNX operators or rewrite using standard ops
```python
# Example of registering a custom op
from torch.onnx.symbolic_helper import parse_args
@parse_args('v', 'v', 'i', 'i')
def symbolic_my_op(g, input1, input2, attr1, attr2):
    return g.op("MyOp", input1, input2, attr1_i=attr1, attr2_i=attr2)

# Register the op
torch.onnx.register_custom_op_symbolic('::my_op', symbolic_my_op, opset_version)
```

### 3. Control Flow

**Problem**: Models with if-statements and loops might not export correctly
**Solution**: Use TorchScript with scripting mode instead of tracing
```python
scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, "scripted_model.pt")
```

### 4. Large Models

**Problem**: Very large models may fail to export due to memory issues
**Solution**: Use chunking or model splitting techniques
```python
# Export model in parts if needed
torch.onnx.export(model.encoder, dummy_input, "encoder.onnx")
torch.onnx.export(model.decoder, encoder_output, "decoder.onnx")
```

## Using Exported Models in the CLIP HAR App

Our application provides adapters for different model formats:

```python
# Select model format in Streamlit app
model_format = st.selectbox(
    "Model Format",
    ["PyTorch", "ONNX", "TensorRT"]
)

# Load appropriate model adapter
if model_format == "PyTorch":
    model = PyTorchAdapter("models/model.pt")
elif model_format == "ONNX":
    model = ONNXAdapter("models/model.onnx")
elif model_format == "TensorRT":
    model = TensorRTAdapter("models/model.trt")

# Use model for inference
result = model.predict(input_data)
```

## Docker Integration

Our Docker setup includes support for these different formats:

- ONNX Runtime is pre-installed in the app container
- TensorRT support is available via the NVIDIA container
- The app automatically selects the appropriate runtime

## Further Resources

- [ONNX Official Documentation](https://onnx.ai/)
- [TensorRT Documentation](https://developer.nvidia.com/tensorrt)
- [PyTorch to ONNX Tutorial](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
- [ONNX Runtime GitHub](https://github.com/microsoft/onnxruntime)
- [Torch-TensorRT Documentation](https://pytorch.org/TensorRT/)

In the next guide, we'll explore how to set up the inference API to serve your exported models. 