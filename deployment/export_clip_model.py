import os
import argparse
import torch
from pathlib import Path

from CLIP_HAR_PROJECT.models.clip_model import CLIPLabelRetriever
from CLIP_HAR_PROJECT.configs import get_config
from CLIP_HAR_PROJECT.deployment.export import (
    export_to_onnx,
    export_to_tensorrt,
    validate_exported_model,
    benchmark_model,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export CLIP HAR model to ONNX/TensorRT"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained PyTorch model",
    )
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to the model config file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="exports",
        help="Directory to save exported models",
    )
    parser.add_argument(
        "--export_format",
        type=str,
        choices=["onnx", "tensorrt", "all"],
        default="all",
        help="Format to export the model to",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="1,3,224,224",
        help="Input shape in format 'batch,channels,height,width'",
    )
    parser.add_argument(
        "--opset_version", type=int, default=13, help="ONNX opset version"
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "int8"],
        default="fp16",
        help="Precision for TensorRT",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate exported models against PyTorch model",
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Benchmark exported models"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Maximum batch size for TensorRT engine",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Parse input shape
    input_shape = tuple(map(int, args.input_shape.split(",")))

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load configuration
    config = get_config(args.config_path)

    # Load model
    print(f"Loading model from {args.model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = CLIPLabelRetriever(
        model_name=config.model.clip_model_name, num_classes=config.model.num_classes
    )

    checkpoint = torch.load(args.model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print(f"Model loaded successfully")

    # File paths
    model_name = Path(args.model_path).stem
    onnx_path = os.path.join(args.output_dir, f"{model_name}.onnx")
    tensorrt_path = os.path.join(args.output_dir, f"{model_name}.trt")

    # Export to ONNX
    if args.export_format in ["onnx", "all"]:
        print("-" * 50)
        print("Exporting to ONNX...")
        export_to_onnx(
            model=model,
            save_path=onnx_path,
            input_shape=input_shape,
            opset_version=args.opset_version,
            verbose=True,
        )

        # Validate ONNX model
        if args.validate:
            print("-" * 50)
            print("Validating ONNX model...")
            is_valid, results = validate_exported_model(
                pytorch_model=model,
                onnx_path=onnx_path,
                input_shape=input_shape,
                device=device,
            )
            print(f"ONNX validation {'successful' if is_valid else 'failed'}")
            print(f"Max absolute difference: {results['max_absolute_diff']:.6f}")
            print(f"Mean absolute difference: {results['mean_absolute_diff']:.6f}")

        # Benchmark ONNX model
        if args.benchmark:
            print("-" * 50)
            print("Benchmarking ONNX model...")
            onnx_results = benchmark_model(
                model_path=onnx_path,
                input_shape=input_shape,
                model_type="onnx",
                device=device,
            )
            print(f"ONNX inference time: {onnx_results['average_inference_ms']:.2f} ms")
            print(f"ONNX throughput: {onnx_results['fps']:.2f} FPS")

    # Export to TensorRT (from ONNX)
    if args.export_format in ["tensorrt", "all"] and torch.cuda.is_available():
        # Check if ONNX file exists
        if not os.path.exists(onnx_path) and args.export_format == "tensorrt":
            print("Exporting to ONNX first...")
            export_to_onnx(
                model=model,
                save_path=onnx_path,
                input_shape=input_shape,
                opset_version=args.opset_version,
                verbose=True,
            )

        print("-" * 50)
        print("Exporting to TensorRT...")
        export_to_tensorrt(
            onnx_path=onnx_path,
            save_path=tensorrt_path,
            precision=args.precision,
            max_batch_size=args.batch_size,
            verbose=True,
        )

        # Benchmark TensorRT model
        if args.benchmark:
            print("-" * 50)
            print("Benchmarking TensorRT model...")
            trt_results = benchmark_model(
                model_path=tensorrt_path,
                input_shape=input_shape,
                model_type="tensorrt",
                device="cuda",
            )
            print(
                f"TensorRT inference time: {trt_results['average_inference_ms']:.2f} ms"
            )
            print(f"TensorRT throughput: {trt_results['fps']:.2f} FPS")

    print("-" * 50)
    print("Export process completed successfully!")

    # Compare benchmark results if both models are benchmarked
    if args.benchmark and args.export_format == "all" and torch.cuda.is_available():
        print("-" * 50)
        print("Performance comparison:")
        print(
            f"PyTorch: {benchmark_model(args.model_path, input_shape, 'pytorch', device=device)['average_inference_ms']:.2f} ms"
        )
        print(f"ONNX   : {onnx_results['average_inference_ms']:.2f} ms")
        print(f"TensorRT: {trt_results['average_inference_ms']:.2f} ms")
        print(
            f"Speed improvement (TensorRT vs PyTorch): {trt_results['fps'] / benchmark_model(args.model_path, input_shape, 'pytorch', device=device)['fps']:.2f}x"
        )


if __name__ == "__main__":
    main()
