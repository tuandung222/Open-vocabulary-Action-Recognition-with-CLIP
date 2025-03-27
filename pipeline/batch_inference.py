#!/usr/bin/env python
# Batch inference script for HAR classification using CLIP

import os
import sys
import argparse
import json
import logging
import numpy as np
import pandas as pd
import torch
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from tqdm import tqdm

# Import project modules
from CLIP_HAR_PROJECT.pipeline.inference_pipeline import InferencePipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("batch_inference")


def chunk_list(input_list: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size."""
    return [
        input_list[i : i + chunk_size] for i in range(0, len(input_list), chunk_size)
    ]


def process_chunk(
    chunk: List[str],
    model_path: str,
    model_type: str,
    model_name: Optional[str],
    labels: Optional[List[str]],
    device: str,
    image_size: int,
    confidence_threshold: float,
) -> List[Dict[str, Any]]:
    """Process a chunk of images using the inference pipeline."""
    # Create inference pipeline
    pipeline = InferencePipeline(
        model_path=model_path,
        model_type=model_type,
        model_name=model_name,
        labels=labels,
        device=device,
        image_size=image_size,
        class_confidence_threshold=confidence_threshold,
    )

    results = []
    for image_path in chunk:
        try:
            # Get relative path for cleaner output
            result = pipeline.predict(image_path)
            results.append((image_path, result))
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            results.append((image_path, {"error": str(e)}))

    # Clean up
    pipeline.cleanup()

    return results


class BatchInferencePipeline:
    """
    Pipeline for processing large batches of images or videos efficiently.

    This pipeline handles:
    1. Parallel processing of input data
    2. Resource management
    3. Results aggregation and analysis

    Supports:
    - Directory of images
    - Directory of videos
    - CSV file with paths
    - JSON file with paths
    """

    def __init__(
        self,
        model_path: str,
        model_type: str = "pytorch",
        model_name: Optional[str] = "openai/clip-vit-base-patch16",
        labels_file: Optional[str] = None,
        batch_size: int = 16,
        num_workers: int = 4,
        device: Optional[str] = None,
        image_size: int = 224,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize the batch inference pipeline.

        Args:
            model_path: Path to the model file
            model_type: Type of model (pytorch, mlflow, onnx, tensorrt)
            model_name: Name of the base CLIP model (for PyTorch model)
            labels_file: Path to file with class labels (for ONNX/TensorRT models)
            batch_size: Number of inputs to process in each batch
            num_workers: Number of parallel workers
            device: Device to run inference on
            image_size: Size of the input image
            confidence_threshold: Threshold for class confidence
        """
        self.model_path = model_path
        self.model_type = model_type
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = min(num_workers, os.cpu_count() or 1)
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold

        # Load labels if provided
        self.labels = None
        if labels_file:
            with open(labels_file, "r") as f:
                self.labels = json.load(f)

        # Determine device
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # If using multiple GPUs, we'll distribute work across them
        self.num_gpus = torch.cuda.device_count() if self.device == "cuda" else 0

        logger.info(f"Initialized batch inference pipeline with:")
        logger.info(f"  - Model: {model_path} ({model_type})")
        logger.info(f"  - Device: {self.device} ({self.num_gpus} GPUs available)")
        logger.info(f"  - Workers: {self.num_workers}")
        logger.info(f"  - Batch size: {self.batch_size}")

    def _get_device_for_worker(self, worker_id: int) -> str:
        """Get the device to use for a specific worker."""
        if self.device == "cuda" and self.num_gpus > 0:
            # Distribute workers across available GPUs
            gpu_id = worker_id % self.num_gpus
            return f"cuda:{gpu_id}"
        else:
            return self.device

    def process_image_directory(
        self,
        input_dir: str,
        output_path: str,
        recursive: bool = False,
        image_extensions: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".webp"],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process all images in a directory using parallel workers.

        Args:
            input_dir: Input directory path
            output_path: Path to save results
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
                os.path.join(input_dir, f)
                for f in os.listdir(input_dir)
                if os.path.isfile(os.path.join(input_dir, f))
                and any(f.lower().endswith(ext) for ext in image_extensions)
            ]

        # Sort files for deterministic ordering
        image_files.sort()

        logger.info(f"Found {len(image_files)} images in {input_dir}")

        # Process images in parallel
        return self._process_files_in_parallel(image_files, output_path, input_dir)

    def process_video_directory(
        self,
        input_dir: str,
        output_path: str,
        recursive: bool = False,
        video_extensions: List[str] = [".mp4", ".avi", ".mov", ".mkv"],
        frame_interval: int = 1,
        save_processed_videos: bool = False,
        output_video_dir: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process all videos in a directory using parallel workers.

        Args:
            input_dir: Input directory path
            output_path: Path to save results
            recursive: Whether to search subdirectories
            video_extensions: List of video file extensions to process
            frame_interval: Process every N frames
            save_processed_videos: Whether to save processed videos
            output_video_dir: Directory to save processed videos

        Returns:
            Dictionary mapping file paths to prediction results
        """
        # Ensure directory exists
        if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
            raise ValueError(f"Directory not found: {input_dir}")

        # Find video files
        video_files = []
        if recursive:
            for root, _, files in os.walk(input_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in video_extensions):
                        video_files.append(os.path.join(root, file))
        else:
            video_files = [
                os.path.join(input_dir, f)
                for f in os.listdir(input_dir)
                if os.path.isfile(os.path.join(input_dir, f))
                and any(f.lower().endswith(ext) for ext in video_extensions)
            ]

        # Sort files for deterministic ordering
        video_files.sort()

        logger.info(f"Found {len(video_files)} videos in {input_dir}")

        # Create output video directory if needed
        if save_processed_videos:
            if output_video_dir is None:
                output_video_dir = os.path.join(
                    os.path.dirname(output_path), "processed_videos"
                )
            os.makedirs(output_video_dir, exist_ok=True)

        # Process each video sequentially (videos are already parallelized internally)
        all_results = {}
        for video_file in tqdm(video_files, desc="Processing videos"):
            try:
                # Create relative path for cleaner output
                rel_path = os.path.relpath(video_file, input_dir)

                # Create output path for processed video if needed
                video_output_path = None
                if save_processed_videos:
                    video_basename = os.path.basename(video_file)
                    video_output_path = os.path.join(
                        output_video_dir, f"processed_{video_basename}"
                    )

                # Create inference pipeline
                pipeline = InferencePipeline(
                    model_path=self.model_path,
                    model_type=self.model_type,
                    model_name=self.model_name,
                    labels=self.labels,
                    device=self.device,
                    image_size=self.image_size,
                    class_confidence_threshold=self.confidence_threshold,
                )

                # Process video
                results = pipeline.predict_video(
                    video_path=video_file,
                    output_path=video_output_path,
                    frame_interval=frame_interval,
                    show_display=False,
                )

                # Clean up
                pipeline.cleanup()

                # Extract summary statistics
                class_counts = {}
                for frame_result in results:
                    if "predicted_class" in frame_result:
                        cls = frame_result["predicted_class"]
                        class_counts[cls] = class_counts.get(cls, 0) + 1

                total_frames = len(results)
                top_class = (
                    max(class_counts.items(), key=lambda x: x[1])[0]
                    if class_counts
                    else None
                )
                top_class_percentage = (
                    max(class_counts.values()) / total_frames * 100
                    if class_counts
                    else 0
                )

                # Store summary results
                video_summary = {
                    "file_path": video_file,
                    "frames_analyzed": total_frames,
                    "top_class": top_class,
                    "top_class_percentage": top_class_percentage,
                    "class_distribution": class_counts,
                    "frame_results": results[
                        :5
                    ],  # Store only first 5 frame results to save space
                    "processed_video_path": video_output_path
                    if save_processed_videos
                    else None,
                }

                all_results[rel_path] = video_summary

            except Exception as e:
                logger.error(f"Error processing video {video_file}: {e}")
                all_results[video_file] = {"error": str(e)}

        # Save results
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"Results saved to {output_path}")

        # Create summary CSV
        summary_df = pd.DataFrame(
            [
                {
                    "file": k,
                    "frames_analyzed": v.get("frames_analyzed", 0)
                    if not isinstance(v, str) and not "error" in v
                    else 0,
                    "top_class": v.get("top_class", "")
                    if not isinstance(v, str) and not "error" in v
                    else "",
                    "top_class_percentage": v.get("top_class_percentage", 0)
                    if not isinstance(v, str) and not "error" in v
                    else 0,
                    "error": v.get("error", "")
                    if isinstance(v, dict) and "error" in v
                    else "",
                }
                for k, v in all_results.items()
            ]
        )

        csv_path = os.path.splitext(output_path)[0] + "_summary.csv"
        summary_df.to_csv(csv_path, index=False)

        return all_results

    def process_from_csv(
        self,
        csv_path: str,
        output_path: str,
        path_column: str = "path",
        base_dir: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process images listed in a CSV file.

        Args:
            csv_path: Path to CSV file
            output_path: Path to save results
            path_column: Name of column containing file paths
            base_dir: Base directory to prepend to relative paths

        Returns:
            Dictionary mapping file paths to prediction results
        """
        # Load CSV file
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")

        # Check if path column exists
        if path_column not in df.columns:
            raise ValueError(f"Column '{path_column}' not found in CSV file")

        # Extract file paths
        file_paths = df[path_column].tolist()

        # Prepend base directory if provided
        if base_dir:
            file_paths = [os.path.join(base_dir, p) for p in file_paths]

        logger.info(f"Found {len(file_paths)} files in CSV")

        # Process files in parallel
        return self._process_files_in_parallel(file_paths, output_path)

    def _process_files_in_parallel(
        self, file_paths: List[str], output_path: str, base_dir: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process a list of files in parallel.

        Args:
            file_paths: List of file paths to process
            output_path: Path to save results
            base_dir: Base directory for extracting relative paths

        Returns:
            Dictionary mapping file paths to prediction results
        """
        # Split files into chunks
        chunks = chunk_list(file_paths, self.batch_size)

        logger.info(
            f"Processing {len(file_paths)} files in {len(chunks)} batches with {self.num_workers} workers"
        )

        # Create directory for output
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Process chunks in parallel
        results = {}
        total_chunks = len(chunks)
        with tqdm(total=total_chunks, desc="Processing batches") as pbar:
            # Use ThreadPoolExecutor for I/O-bound operations
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.num_workers
            ) as executor:
                # Submit all tasks
                future_to_chunk = {
                    executor.submit(
                        process_chunk,
                        chunk,
                        self.model_path,
                        self.model_type,
                        self.model_name,
                        self.labels,
                        self._get_device_for_worker(i % self.num_workers),
                        self.image_size,
                        self.confidence_threshold,
                    ): i
                    for i, chunk in enumerate(chunks)
                }

                # Process completed tasks
                for future in concurrent.futures.as_completed(future_to_chunk):
                    chunk_results = future.result()

                    # Add results to final dictionary
                    for file_path, result in chunk_results:
                        # Use relative path if base_dir is provided
                        key = (
                            os.path.relpath(file_path, base_dir)
                            if base_dir
                            else file_path
                        )
                        results[key] = result

                    # Update progress bar
                    pbar.update(1)

        # Save results
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_path}")

        # Create summary CSV
        summary_df = pd.DataFrame(
            [
                {
                    "file": k,
                    "predicted_class": v.get("predicted_class", "")
                    if not isinstance(v, str) and not "error" in v
                    else "",
                    "confidence": v.get("confidence", 0)
                    if not isinstance(v, str) and not "error" in v
                    else 0,
                    "error": v.get("error", "")
                    if isinstance(v, dict) and "error" in v
                    else "",
                }
                for k, v in results.items()
            ]
        )

        csv_path = os.path.splitext(output_path)[0] + "_summary.csv"
        summary_df.to_csv(csv_path, index=False)

        return results

    def analyze_results(
        self, results_path: str, output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze batch inference results and generate visualizations.

        Args:
            results_path: Path to JSON results file
            output_dir: Directory to save analysis artifacts

        Returns:
            Dictionary with analysis results
        """
        # Load results
        with open(results_path, "r") as f:
            results = json.load(f)

        # Set output directory
        if output_dir is None:
            output_dir = os.path.dirname(results_path)
        os.makedirs(output_dir, exist_ok=True)

        # Extract predictions
        predictions = []
        errors = []

        for file_path, result in results.items():
            if isinstance(result, dict) and "error" in result:
                errors.append((file_path, result["error"]))
            elif isinstance(result, dict) and "predicted_class" in result:
                predictions.append(
                    {
                        "file_path": file_path,
                        "predicted_class": result["predicted_class"],
                        "confidence": result["confidence"],
                    }
                )

        # Convert to DataFrame
        if predictions:
            df = pd.DataFrame(predictions)

            # Generate class distribution
            class_counts = df["predicted_class"].value_counts()
            class_percentages = class_counts / len(df) * 100

            # Calculate average confidence per class
            avg_confidence = df.groupby("predicted_class")["confidence"].mean()

            # Combine statistics
            stats_df = pd.DataFrame(
                {
                    "count": class_counts,
                    "percentage": class_percentages,
                    "avg_confidence": avg_confidence,
                }
            )

            # Save statistics
            stats_path = os.path.join(output_dir, "class_statistics.csv")
            stats_df.to_csv(stats_path)

            # Generate summary
            summary = {
                "total_processed": len(results),
                "successful_predictions": len(predictions),
                "errors": len(errors),
                "unique_classes": len(class_counts),
                "most_common_class": class_counts.index[0]
                if len(class_counts) > 0
                else None,
                "most_common_class_count": class_counts.iloc[0]
                if len(class_counts) > 0
                else 0,
                "most_common_class_percentage": class_percentages.iloc[0]
                if len(class_percentages) > 0
                else 0,
                "highest_confidence_class": avg_confidence.idxmax()
                if len(avg_confidence) > 0
                else None,
                "highest_confidence": avg_confidence.max()
                if len(avg_confidence) > 0
                else 0,
            }

            # Save summary
            summary_path = os.path.join(output_dir, "analysis_summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            # Generate visualizations if matplotlib is available
            try:
                import matplotlib.pyplot as plt

                # Class distribution plot
                plt.figure(figsize=(12, 6))
                class_counts.plot(kind="bar")
                plt.title("Class Distribution")
                plt.xlabel("Class")
                plt.ylabel("Count")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=300)
                plt.close()

                # Confidence distribution plot
                plt.figure(figsize=(12, 6))
                avg_confidence.plot(kind="bar")
                plt.title("Average Confidence by Class")
                plt.xlabel("Class")
                plt.ylabel("Confidence")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(output_dir, "confidence_distribution.png"), dpi=300
                )
                plt.close()

                # Add paths to summary
                summary["class_distribution_plot"] = os.path.join(
                    output_dir, "class_distribution.png"
                )
                summary["confidence_distribution_plot"] = os.path.join(
                    output_dir, "confidence_distribution.png"
                )
            except ImportError:
                logger.warning(
                    "Matplotlib not available. Skipping visualization generation."
                )

            logger.info(f"Analysis saved to {output_dir}")

            return summary
        else:
            logger.warning("No valid predictions found in results.")
            return {"error": "No valid predictions found"}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run batch inference for HAR classification"
    )

    # Model arguments
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model file"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="pytorch",
        choices=["pytorch", "mlflow", "onnx", "tensorrt"],
        help="Type of model",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/clip-vit-base-patch16",
        help="Name of the base CLIP model (for PyTorch models)",
    )
    parser.add_argument(
        "--labels_file",
        type=str,
        default=None,
        help="Path to JSON file with class labels (for ONNX/TensorRT models)",
    )

    # Input arguments (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image_dir", type=str, help="Directory containing images"
    )
    input_group.add_argument(
        "--video_dir", type=str, help="Directory containing videos"
    )
    input_group.add_argument(
        "--csv_path", type=str, help="Path to CSV file containing file paths"
    )

    # Output arguments
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save results JSON file"
    )

    # Processing arguments
    parser.add_argument(
        "--recursive", action="store_true", help="Recursively search subdirectories"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of items to process in each batch",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of parallel workers"
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=5,
        help="Process every N frames (for videos)",
    )
    parser.add_argument(
        "--save_processed_videos", action="store_true", help="Save processed videos"
    )
    parser.add_argument(
        "--csv_path_column",
        type=str,
        default="path",
        help="Name of column containing file paths (for CSV input)",
    )
    parser.add_argument(
        "--csv_base_dir",
        type=str,
        default=None,
        help="Base directory to prepend to relative paths (for CSV input)",
    )
    parser.add_argument(
        "--image_size", type=int, default=224, help="Size of the input image"
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Threshold for class confidence",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (cpu, cuda)",
    )
    parser.add_argument(
        "--analyze_results",
        action="store_true",
        help="Analyze results after processing",
    )

    return parser.parse_args()


def main():
    """Main function to run the batch inference pipeline."""
    # Parse command line arguments
    args = parse_args()

    try:
        # Create batch inference pipeline
        pipeline = BatchInferencePipeline(
            model_path=args.model_path,
            model_type=args.model_type,
            model_name=args.model_name,
            labels_file=args.labels_file,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            image_size=args.image_size,
            confidence_threshold=args.confidence_threshold,
        )

        # Run inference based on input type
        if args.image_dir:
            # Process directory of images
            results = pipeline.process_image_directory(
                input_dir=args.image_dir,
                output_path=args.output_path,
                recursive=args.recursive,
            )

        elif args.video_dir:
            # Process directory of videos
            results = pipeline.process_video_directory(
                input_dir=args.video_dir,
                output_path=args.output_path,
                recursive=args.recursive,
                frame_interval=args.frame_interval,
                save_processed_videos=args.save_processed_videos,
            )

        elif args.csv_path:
            # Process files listed in CSV
            results = pipeline.process_from_csv(
                csv_path=args.csv_path,
                output_path=args.output_path,
                path_column=args.csv_path_column,
                base_dir=args.csv_base_dir,
            )

        # Analyze results if requested
        if args.analyze_results:
            analysis = pipeline.analyze_results(args.output_path)
            logger.info("Analysis summary:")
            for key, value in analysis.items():
                if not isinstance(value, (str, int, float, bool, type(None))):
                    continue
                logger.info(f"  {key}: {value}")

        return 0

    except Exception as e:
        logger.exception(f"Error in batch inference pipeline: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
