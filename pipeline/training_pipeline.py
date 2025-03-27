#!/usr/bin/env python
# Training pipeline for HAR classification using CLIP

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist

# Import project modules
from configs.default import get_config
from data.augmentation import HARDataAugmentation
# from datasets import load_dataset
from data.preprocessing import (
    create_class_distribution_visualizations,
    get_class_mappings,
    prepare_har_dataset,
    visualize_samples,
)
from deployment.export import (
    benchmark_model,
    export_to_onnx,
    export_to_tensorrt,
    validate_exported_model,
)
from evaluation.evaluator import ClassificationEvaluator, get_evaluator

# Import evaluation modules
from evaluation.metrics import (
    compute_classification_report,
    compute_confusion_matrix,
    compute_metrics,
)
from evaluation.visualization import (
    create_evaluation_report,
    plot_accuracy_per_class,
    plot_confusion_matrix,
)
from mlops.tracking import (
    create_tracker,
    log_artifact,
    log_confusion_matrix,
    log_metrics,
    log_model,
    log_model_params,
    setup_mlflow,
)
from models.clip_model import (
    CLIPLabelRetriever,
    freeze_clip_parameters,
    get_labels_from_dataset,
    print_trainable_parameters,
)
from training.distributed import (
    cleanup_distributed,
    setup_distributed_environment,
)
from training.trainer import (
    DistributedTrainer,
    TrainingConfig,
    get_trainer,
)
from utils.config import Config
from utils.distributed import get_rank, get_world_size, is_main_process
from utils.mlflow import (
    log_artifact,
    log_metrics,
    log_model,
    log_params,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("training_pipeline")


class TrainingPipeline:
    """
    End-to-end training pipeline for HAR classification using CLIP.

    This pipeline handles:
    1. Data preparation and augmentation
    2. Model configuration
    3. Distributed training setup
    4. Model training and evaluation
    5. Model export to deployment formats
    6. MLflow tracking and artifacts
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        distributed_mode: str = "none",
        use_mlflow: bool = True,
        use_wandb: bool = True,
        experiment_name: Optional[str] = None,
        config: Optional[Any] = None,
        **kwargs,
    ):
        """
        Initialize the training pipeline.

        Args:
            config_path: Path to configuration file (YAML/JSON)
            distributed_mode: Training mode (none, ddp, fsdp)
            use_mlflow: Whether to use MLflow for tracking
            use_wandb: Whether to use wandb for tracking
            experiment_name: Experiment name for tracking
            config: Configuration object (can be provided directly instead of config_path)
            **kwargs: Additional configuration overrides
        """
        # Load configuration
        if config is not None:
            self.config = config
        else:
            self.config = get_config(distributed_mode)
            if config_path:
                self.config = self.config.merge_from_file(config_path)

        # Update config with kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Initialize distributed environment
        is_distributed = distributed_mode != "none"
        self.rank = 0
        self.world_size = 1
        if is_distributed:
            self.rank, self.world_size = setup_distributed_environment(distributed_mode)
        self.is_main_process = not is_distributed or self.rank == 0

        # Setup experiment tracking (only on main process)
        self.use_mlflow = use_mlflow
        self.use_wandb = use_wandb
        self.experiment_name = experiment_name or "clip_har_training"
        self.tracker = None

        if self.is_main_process and (use_mlflow or use_wandb):
            self.tracker = create_tracker(
                use_mlflow=use_mlflow,
                use_wandb=use_wandb,
                experiment_name=self.experiment_name,
                mlflow_tracking_uri=self.config.mlflow.tracking_uri
                if hasattr(self.config, "mlflow")
                else None,
                mlflow_artifacts_dir=self.config.mlflow.artifacts_dir
                if hasattr(self.config, "mlflow")
                else "./mlruns",
                wandb_project=self.config.wandb.project_name
                if hasattr(self.config, "wandb")
                else "clip_har",
                wandb_group=self.config.wandb.group_name
                if hasattr(self.config, "wandb")
                else None,
                wandb_entity=self.config.wandb.entity
                if hasattr(self.config, "wandb")
                else None,
                config={
                    "model": self.config.model.__dict__
                    if hasattr(self.config, "model")
                    else {},
                    "training": self.config.training.__dict__
                    if hasattr(self.config, "training")
                    else {},
                    "data": self.config.data.__dict__
                    if hasattr(self.config, "data")
                    else {},
                },
            )
            # Start the tracking run
            self.tracker.start_run(
                run_name=f"{self.experiment_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

        # Initialize other attributes
        self.tokenizer = None
        self.image_processor = None
        self.datasets = None
        self.class_names = None
        self.model = None
        self.trainer = None
        self.training_results = None
        self.export_paths = {}

    def __del__(self):
        """Cleanup when the pipeline is destroyed."""
        # End tracking run if active
        if hasattr(self, "tracker") and self.tracker:
            self.tracker.end_run()

        # Clean up any distributed environment
        if hasattr(self, "is_distributed") and self.is_distributed:
            cleanup_distributed()

    def prepare_data(
        self,
        augmentation_strength: str = "medium",
        use_action_specific_augmentation: bool = False,
        use_mix_augmentation: bool = False,
        mix_type: str = "mixup",
        visualize: bool = True,
    ) -> Tuple[Dict, List[str]]:
        """
        Prepare and preprocess datasets.

        Args:
            augmentation_strength: Strength of augmentation (light, medium, strong)
            use_action_specific_augmentation: Whether to use action-specific augmentation
            use_mix_augmentation: Whether to use mix augmentation
            mix_type: Type of mix augmentation (mixup, cutmix, both)
            visualize: Whether to visualize samples and class distribution

        Returns:
            Tuple of (datasets, class_names)
        """
        from transformers import CLIPImageProcessor, CLIPTokenizerFast

        logger.info("Preparing datasets...")

        # Load tokenizer and image processor
        self.tokenizer = CLIPTokenizerFast.from_pretrained(self.config.model.model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.config.model.model_name
        )

        # Get data augmentation transforms
        transforms = None
        if augmentation_strength != "none":
            transforms = HARDataAugmentation.get_train_transforms(
                image_size=self.config.data.image_size,
                augmentation_strength=augmentation_strength,
            )
            logger.info(f"Using {augmentation_strength} augmentation strength")

        # Get mix augmentation config
        mix_config = None
        if use_mix_augmentation:
            mix_config = HARDataAugmentation.get_mix_augmentation(
                mix_type=mix_type, alpha=0.2
            )
            logger.info(f"Using {mix_type} augmentation")

        # Prepare dataset
        self.datasets, self.class_names = prepare_har_dataset(
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            val_ratio=self.config.data.val_ratio,
            test_ratio=self.config.data.test_ratio,
            seed=self.config.data.seed,
            transforms=transforms,
            use_mix_augmentation=use_mix_augmentation,
            mix_config=mix_config,
        )

        # Visualize samples and class distribution (main process only)
        if self.is_main_process and visualize:
            logger.info(f"Class names: {self.class_names}")
            _, id2string, _ = get_class_mappings(None)

            # Debug dataset structure
            logger.info("Debugging dataset structure:")
            try:
                # Get information about the train dataset
                train_sample = self.datasets["train"][0]
                logger.info(f"Train dataset sample keys: {train_sample.keys()}")
                
                # Check actual features in the dataset
                if hasattr(self.datasets["train"], "features"):
                    logger.info(f"Train dataset features: {self.datasets['train'].features}")
                if hasattr(self.datasets["train"], "column_names"):
                    logger.info(f"Train dataset columns: {self.datasets['train'].column_names}")
                    
                # Try to print first few samples' labels
                for i in range(min(3, len(self.datasets["train"]))):
                    sample = self.datasets["train"][i]
                    label_info = {}
                    for key in ["labels", "label_id", "label"]:
                        if key in sample:
                            label_info[key] = sample[key]
                    logger.info(f"Sample {i} labels: {label_info}")
            except Exception as e:
                logger.error(f"Error debugging dataset: {e}")

            # Visualize samples
            try:
                visualize_samples(
                    self.datasets["train"],
                    id2string,
                    save_dir=self.config.training.output_dir,
                )
            except Exception as e:
                logger.error(f"Error visualizing samples: {e}")
                import traceback
                traceback.print_exc()

            # Visualize class distribution with error handling
            try:
                # For visualization, access the raw dataset before transformation if possible
                original_dataset = self.datasets["train"]
                
                # Add debugging information
                if hasattr(original_dataset, '_data') and hasattr(original_dataset._data, '_info'):
                    logger.info(f"Dataset info: {original_dataset._data._info}")
                
                # Try visualization with original dataset first
                vis_results = create_class_distribution_visualizations(
                    original_dataset, id2string, save_dir=self.config.training.output_dir
                )
                if vis_results is None or vis_results[0] is None:
                    logger.warning("Class distribution visualization failed with train dataset - trying with DatasetDict")
                    # Try with whole dataset dict as fallback
                    vis_results = create_class_distribution_visualizations(
                        self.datasets, id2string, save_dir=self.config.training.output_dir
                    )
                    if vis_results is None or vis_results[0] is None:
                        logger.warning("Class distribution visualization failed with DatasetDict - skipping visualizations")
            except Exception as e:
                logger.warning(f"Error creating class distribution visualizations: {e}")
                logger.warning("Continuing without visualizations")

        logger.info(f"Prepared {len(self.datasets['train'])} training samples")
        logger.info(f"Prepared {len(self.datasets['val'])} validation samples")
        logger.info(f"Prepared {len(self.datasets['test'])} test samples")

        return self.datasets, self.class_names

    def setup_model(self) -> torch.nn.Module:
        """
        Configure and setup the model.

        Returns:
            The configured model
        """
        logger.info("Setting up model...")

        # Ensure we have class names
        if self.class_names is None:
            if self.datasets is not None:
                self.class_names = get_labels_from_dataset(self.datasets["train"])
            else:
                raise ValueError("Datasets not prepared. Call prepare_data first.")

        # Create model
        self.model = CLIPLabelRetriever.from_pretrained(
            self.config.model.model_name,
            labels=self.class_names,
            prompt_template=self.config.model.prompt_template,
        )

        # Freeze/unfreeze parameters
        freeze_clip_parameters(
            self.model,
            unfreeze_visual_encoder=self.config.model.unfreeze_visual_encoder,
            unfreeze_text_encoder=self.config.model.unfreeze_text_encoder,
        )

        # Print trainable parameters (main process only)
        if self.is_main_process:
            print_trainable_parameters(self.model)

        return self.model

    def train(self) -> Dict[str, Any]:
        """
        Train the model on the prepared dataset.

        Returns:
            Dictionary with training results
        """
        logger.info("Starting training...")

        # Ensure we have a model
        if self.model is None:
            raise ValueError("Model not set up. Call setup_model first.")

        # Create trainer configuration
        training_config = TrainingConfig(
            max_epochs=self.config.training.max_epochs,
            lr=self.config.training.lr,
            betas=self.config.training.betas,
            weight_decay=self.config.training.weight_decay,
            num_warmup_steps=self.config.training.num_warmup_steps,
            train_log_interval=self.config.training.train_log_interval,
            val_log_interval=self.config.training.val_log_interval,
            max_patience=self.config.training.max_patience,
            output_dir=self.config.training.output_dir,
            batch_size=self.config.training.batch_size,
            eval_batch_size=self.config.training.eval_batch_size,
            num_workers=self.config.training.num_workers,
            mixed_precision=self.config.training.mixed_precision,
            distributed_mode=self.config.training.distributed_mode,
        )

        # Create trainer
        self.trainer = get_trainer(
            config=training_config,
            model=self.model,
            train_dataset=self.datasets["train"],
            val_dataset=self.datasets["val"],
            tokenizer=self.tokenizer,
            labels=self.class_names,
            logger=self.tracker,
        )

        # Log model parameters if tracking enabled
        if self.is_main_process and self.tracker:
            # Get model parameters
            num_params = sum(p.numel() for p in self.model.parameters())
            num_trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )

            # Log parameters
            self.tracker.log_params(
                {
                    "num_params": num_params,
                    "num_trainable_params": num_trainable_params,
                    "trainable_param_ratio": num_trainable_params / num_params
                    if num_params > 0
                    else 0,
                    "model.name": self.config.model.model_name,
                    "model.unfreeze_visual": self.config.model.unfreeze_visual_encoder,
                    "model.unfreeze_text": self.config.model.unfreeze_text_encoder,
                    "dataset.train_size": len(self.datasets["train"]),
                    "dataset.val_size": len(self.datasets["val"]),
                    "dataset.test_size": len(self.datasets["test"]),
                    "training.batch_size": self.config.training.batch_size,
                    "training.max_epochs": self.config.training.max_epochs,
                    "training.learning_rate": self.config.training.lr,
                    "training.distributed_mode": self.config.training.distributed_mode,
                    "training.world_size": self.world_size,
                }
            )

        # Train model
        results = self.trainer.train()
        self.training_results = results

        # Log best metrics if tracking enabled
        if self.is_main_process and self.tracker and results:
            # Log best metrics
            if "best_metrics" in results:
                self.tracker.log_metrics(results["best_metrics"])

            # Log final checkpoint
            if hasattr(self.trainer, "best_model_path") and os.path.exists(
                self.trainer.best_model_path
            ):
                self.tracker.log_artifact(self.trainer.best_model_path, "checkpoints")

            # Log learning curve
            if "history" in results and len(results["history"]) > 0:
                # Create learning curve plot
                plt.figure(figsize=(10, 6))
                for metric in ["train_loss", "val_loss"]:
                    if metric in results["history"][0]:
                        plt.plot(
                            [h["epoch"] for h in results["history"]],
                            [h[metric] for h in results["history"]],
                            label=metric,
                        )
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                plt.title("Learning Curve")
                plt.grid(True, alpha=0.3)

                # Log figure
                self.tracker.log_figure(plt.gcf(), "learning_curve.png")
                plt.close()

        return results

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the trained model on the test set.

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model on test set...")

        # Ensure we have a model and test dataset
        if self.model is None:
            raise ValueError("Model not available. Call train first.")
        if "test" not in self.datasets:
            raise ValueError("Test dataset not available.")

        # Create evaluator
        evaluator = get_evaluator(
            eval_type="classification",
            dataset=self.datasets["test"],
            class_names=self.class_names,
            batch_size=self.config.training.eval_batch_size,
            num_workers=self.config.training.num_workers,
            device=self.device,
            output_dir=os.path.join(self.config.training.output_dir, "evaluation"),
            collate_fn=getattr(self.trainer, "collate_fn", None),
            visualize_results=True,
            num_visualizations=10,
        )

        # Evaluate model
        eval_metrics = evaluator.evaluate(self.model)

        # Log evaluation metrics to trackers (main process only)
        if self.is_main_process and self.tracker:
            # Log metrics
            self.tracker.log_metrics(
                {
                    f"test_{k}": v
                    for k, v in eval_metrics.items()
                    if isinstance(v, (int, float))
                }
            )

            # Log confusion matrix
            if "predictions" in eval_metrics and "references" in eval_metrics:
                self.tracker.log_confusion_matrix(
                    eval_metrics["references"],
                    eval_metrics["predictions"],
                    self.class_names,
                )

            # Log visualizations if available
            if "visualization_files" in eval_metrics:
                for name, file_path in eval_metrics["visualization_files"].items():
                    self.tracker.log_artifact(file_path, "evaluation")

            logger.info("Logged evaluation results to experiment trackers")

        # Print evaluation metrics (main process only)
        if self.is_main_process:
            logger.info(f"Evaluation metrics: {eval_metrics}")

        return eval_metrics

    def export_model(self, formats: List[str] = ["onnx"]) -> Dict[str, str]:
        """
        Export the trained model to deployment formats.

        Args:
            formats: List of export formats ("onnx", "tensorrt")

        Returns:
            Dictionary mapping format to export path
        """
        logger.info(f"Exporting model to formats: {formats}")

        # Ensure we have a model
        if self.model is None:
            raise ValueError("Model not available. Call train first.")

        # Create export directory
        export_dir = os.path.join(self.config.training.output_dir, "exports")
        os.makedirs(export_dir, exist_ok=True)

        # Export to each format
        self.export_paths = {}

        if "onnx" in formats:
            try:
                logger.info("Exporting to ONNX format...")
                onnx_path = export_to_onnx(
                    model=self.model,
                    output_path=os.path.join(export_dir, "model.onnx"),
                    input_shape=(
                        1,
                        3,
                        self.config.data.image_size,
                        self.config.data.image_size,
                    ),
                    dynamic_axes=True,
                )
                self.export_paths["onnx"] = onnx_path

                # Validate exported model
                is_valid = validate_exported_model(
                    onnx_path,
                    shape=(
                        1,
                        3,
                        self.config.data.image_size,
                        self.config.data.image_size,
                    ),
                )
                logger.info(
                    f"ONNX model validation: {'passed' if is_valid else 'failed'}"
                )

                # Log to MLflow if enabled
                if self.is_main_process and self.tracker:
                    self.tracker.log_artifact(onnx_path, "exports")
            except Exception as e:
                logger.error(f"Error exporting to ONNX: {e}")

        if "tensorrt" in formats:
            try:
                logger.info("Exporting to TensorRT format...")
                # Requires ONNX model first
                if "onnx" not in self.export_paths:
                    logger.info("ONNX model not available, exporting to ONNX first...")
                    onnx_path = export_to_onnx(
                        model=self.model,
                        output_path=os.path.join(export_dir, "model.onnx"),
                        input_shape=(
                            1,
                            3,
                            self.config.data.image_size,
                            self.config.data.image_size,
                        ),
                        dynamic_axes=True,
                    )
                    self.export_paths["onnx"] = onnx_path

                # Export to TensorRT
                tensorrt_path = export_to_tensorrt(
                    onnx_path=self.export_paths["onnx"],
                    output_path=os.path.join(export_dir, "model.engine"),
                    precision="fp16",
                    workspace_size=8,
                )
                self.export_paths["tensorrt"] = tensorrt_path

                # Log to MLflow if enabled
                if self.is_main_process and self.tracker:
                    self.tracker.log_artifact(tensorrt_path, "exports")
            except Exception as e:
                logger.error(f"Error exporting to TensorRT: {e}")

        # Benchmark exported models if requested
        if self.config.export.benchmark and self.export_paths:
            logger.info("Benchmarking exported models...")
            benchmark_results = {}

            # Get sample input
            sample_input = torch.randn(
                1, 3, self.config.data.image_size, self.config.data.image_size
            ).to(self.device)

            # Benchmark PyTorch model
            benchmark_results["pytorch"] = benchmark_model(
                model=self.model, input_data=sample_input, iterations=100, warmup=10
            )

            # Benchmark exported models
            for format_name, model_path in self.export_paths.items():
                try:
                    benchmark_results[format_name] = benchmark_model(
                        model=model_path,
                        input_data=sample_input,
                        model_type=format_name,
                        iterations=100,
                        warmup=10,
                    )
                except Exception as e:
                    logger.error(f"Error benchmarking {format_name} model: {e}")

            # Log benchmark results
            if self.is_main_process and self.tracker:
                # Convert to loggable metrics
                for format_name, result in benchmark_results.items():
                    self.tracker.log_metrics(
                        {
                            f"benchmark_{format_name}_latency_ms": result[
                                "avg_latency_ms"
                            ],
                            f"benchmark_{format_name}_throughput": result["throughput"],
                        }
                    )

                # Create comparison plot
                plt.figure(figsize=(10, 6))
                formats = list(benchmark_results.keys())
                latencies = [benchmark_results[f]["avg_latency_ms"] for f in formats]

                plt.bar(formats, latencies)
                plt.xlabel("Format")
                plt.ylabel("Latency (ms)")
                plt.title("Model Format Comparison")
                plt.grid(True, alpha=0.3)

                # Log figure
                self.tracker.log_figure(plt.gcf(), "benchmark_comparison.png")
                plt.close()

        return self.export_paths

    def run(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline.

        Returns:
            Dictionary with pipeline results
        """
        try:
            # Execute pipeline steps
            print("prepare_data start")
            self.prepare_data()
            print("prepare_data done")
            self.setup_model()
            print("setup_model done")
            self.train()
            print("train done")
            eval_metrics = self.evaluate()
            print("evaluate done")
            # Export model if configured
            if hasattr(self.config, "export") and self.config.export.enabled:
                self.export_model(formats=self.config.export.formats)

            # Combine results
            results = {
                "training_results": self.training_results,
                "evaluation_metrics": eval_metrics,
                "export_paths": self.export_paths,
            }

            return results

        except Exception as e:
            logger.error(f"Error in pipeline: {e}")

            # End tracking run if active
            if self.is_main_process and self.tracker:
                self.tracker.log_params({"pipeline_error": str(e)})
                self.tracker.end_run()

            raise

    def cleanup(self):
        """Cleanup resources used by the pipeline."""
        # End tracking run if active
        if hasattr(self, "tracker") and self.tracker:
            try:
                self.tracker.end_run()
            except Exception as e:
                logger.warning(f"Error ending tracker run: {e}")
                
        # Clean up any distributed environment
        try:
            if hasattr(self, "distributed_mode") and self.distributed_mode != "none":
                cleanup_distributed()
        except Exception as e:
            logger.warning(f"Error cleaning up distributed environment: {e}")
            
        logger.info("Pipeline resources cleaned up")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run training pipeline for HAR classification"
    )

    # Pipeline configuration
    parser.add_argument(
        "--config_path", type=str, default=None, help="Path to configuration file"
    )
    parser.add_argument(
        "--distributed_mode",
        type=str,
        default="none",
        choices=["none", "ddp", "fsdp"],
        help="Distributed training mode",
    )

    # MLflow arguments
    parser.add_argument(
        "--disable_mlflow", action="store_true", help="Disable MLflow tracking"
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None, help="MLflow experiment name"
    )

    # Data arguments
    parser.add_argument(
        "--augmentation_strength",
        type=str,
        default="medium",
        choices=["none", "light", "medium", "strong"],
        help="Strength of data augmentation",
    )

    # Export arguments
    parser.add_argument(
        "--export_formats",
        type=str,
        nargs="+",
        default=["onnx", "tensorrt"],
        help="List of export formats",
    )

    # Override configuration options
    parser.add_argument(
        "--model_name", type=str, default=None, help="CLIP model name or path"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Output directory for artifacts"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Training batch size"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=None, help="Maximum number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")

    return parser.parse_args()


def main():
    """Main function to run the training pipeline."""
    # Parse command line arguments
    args = parse_args()

    # Create configuration overrides
    overrides = {}
    if args.model_name:
        overrides["model.model_name"] = args.model_name
    if args.output_dir:
        overrides["training.output_dir"] = args.output_dir
    if args.batch_size:
        overrides["training.batch_size"] = args.batch_size
    if args.max_epochs:
        overrides["training.max_epochs"] = args.max_epochs
    if args.lr:
        overrides["training.lr"] = args.lr

    # Create and run training pipeline
    pipeline = TrainingPipeline(
        config_path=args.config_path,
        distributed_mode=args.distributed_mode,
        use_mlflow=not args.disable_mlflow,
        use_wandb=True,
        experiment_name=args.experiment_name,
        **overrides,
    )

    try:
        # Run pipeline
        results = pipeline.run()

        # Print results summary
        rank = pipeline.rank
        if rank == 0:  # Only log from main process
            logger.info("Pipeline completed with results:")
            for key, value in results.items():
                if isinstance(value, dict):
                    logger.info(f"{key}:")
                    for k, v in value.items():
                        logger.info(f"  {k}: {v}")
                else:
                    logger.info(f"{key}: {value}")

        return 0

    finally:
        # Clean up resources
        pipeline.cleanup()


if __name__ == "__main__":
    sys.exit(main())
