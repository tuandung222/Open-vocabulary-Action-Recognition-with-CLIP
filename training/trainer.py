import datetime
import gc
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

import evaluate

# Import data utils
from data.preprocessing import collate_fn

# Import distributed training utilities
from training.distributed import (
    DistributedConfig,
    DistributedMode,
    cleanup_distributed,
    get_distributed_sampler,
    is_main_process,
    load_distributed_model,
    save_distributed_model,
    setup_distributed_environment,
    wrap_model_for_distributed,
)


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Basic training parameters
    max_epochs: int = 15
    lr: float = 3e-6
    betas: Tuple[float, float] = (0.9, 0.995)
    weight_decay: float = 0.01
    num_warmup_steps: int = 3
    train_log_interval: int = 10
    val_log_interval: int = 10
    max_patience: int = 5

    # Output directories
    output_dir: str = "checkpoints"

    # Batch sizes
    batch_size: int = 256
    eval_batch_size: int = 128
    num_workers: int = 2

    # Mixed precision training
    mixed_precision: bool = True

    # Distributed training
    distributed_mode: str = "none"  # "none", "ddp", or "fsdp"

    def get(self, key, default=None):
        """Get a configuration value."""
        return getattr(self, key, default)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }

    @classmethod
    def from_dict(cls, config_dict):
        """Create from dictionary."""
        return cls(**config_dict)


class DistributedTrainer:
    """
    Enhanced trainer that supports distributed training.
    Based on the original MyTrainer class but with added distributed training capabilities.
    """

    def __init__(
        self,
        config: Union[TrainingConfig, DictConfig],
        model: torch.nn.Module,
        train_dataset: Any,
        val_dataset: Any,
        tokenizer: Optional[Any] = None,
        labels: Optional[List[str]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        logger: Optional[Any] = None,
        metrics_to_save_best: Optional[List[str]] = ["val/metrics/accuracy"],
        collate_fn: Optional[Callable] = collate_fn,
    ):
        # Store configuration and logger
        self.config = (
            config
            if isinstance(config, TrainingConfig)
            else TrainingConfig.from_dict(config)
        )
        self._logger = logger

        # Setup distributed environment
        self.distributed_config = self._setup_distributed_config()

        # Initialize training components
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.labels = labels
        self.collate_fn = collate_fn

        # Setup output directory
        self.setup_output_dir()

        # Initialize training state
        self.cur_step = -1
        self.cur_epoch = -1
        self.exit_by_patience = False
        self.current_patience = -1
        self.max_patience = config.get("max_patience", math.inf)
        self.best_metrics_values = {**{key: -1 for key in metrics_to_save_best}}
        self.history_metrics = []

        # Build dataloaders
        self.build_dataloaders()

        # Initialize distributed model, optimizers, etc.
        self.setup_distributed_model()

        # Log config and model
        self.log_config_and_model()

    def _setup_distributed_config(self) -> DistributedConfig:
        """Setup distributed configuration based on the training config."""
        mode_map = {
            "none": DistributedMode.NONE,
            "ddp": DistributedMode.DDP,
            "fsdp": DistributedMode.FSDP,
        }

        mode = mode_map.get(self.config.distributed_mode.lower(), DistributedMode.NONE)

        dist_config = DistributedConfig(
            mode=mode, mixed_precision=self.config.mixed_precision
        )

        # Setup the distributed environment
        return setup_distributed_environment(dist_config)

    def setup_distributed_model(self):
        """Setup model for distributed training."""
        # Wrap model for distributed training
        self.model = wrap_model_for_distributed(self.model, self.distributed_config)

        # Only initialize optimizers/schedulers during actual training
        self.optimizer = None
        self.scheduler = None

    def build_dataloaders(self):
        """Build dataloaders with distributed samplers if necessary."""
        # Get distributed samplers
        train_sampler = get_distributed_sampler(
            self.train_dataset, self.distributed_config, shuffle=True
        )

        val_sampler = get_distributed_sampler(
            self.val_dataset, self.distributed_config, shuffle=False
        )

        # Create dataloaders
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=(train_sampler is None),
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn,
            sampler=train_sampler,
            pin_memory=True,
        )

        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn,
            sampler=val_sampler,
            pin_memory=True,
        )

    def setup_output_dir(self):
        """Setup the output directory for saving checkpoints."""
        if self._logger is None or not is_main_process(self.distributed_config):
            return

        # Get project, group, and experiment names from logger
        self.project_name = getattr(self._logger, "project", "project")
        self.group_name = getattr(self._logger, "group", "group")
        self.experiment_name = getattr(self._logger, "name", "experiment")

        # Create output directory
        output_dir = self.config.get("output_dir", "checkpoints")
        prefix = (
            f"{output_dir}/{self.project_name}/{self.group_name}/{self.experiment_name}"
        )

        if not os.path.exists(prefix):
            os.makedirs(prefix)

        # Save config
        with open(f"{prefix}/config.yaml", "w") as f:
            if isinstance(self.config, DictConfig):
                OmegaConf.save(self.config, f)
            else:
                OmegaConf.save(OmegaConf.create(self.config.to_dict()), f)

        self.checkpoint_prefix = prefix

    def log_config_and_model(self):
        """Log config and model information."""
        if not is_main_process(self.distributed_config):
            return

        print(f"Distributed mode: {self.distributed_config.mode}")
        print(f"World size: {self.distributed_config.world_size}")
        print(f"Rank: {self.distributed_config.rank}")
        print(f"Local rank: {self.distributed_config.local_rank}")
        print(f"Training config: {self.config}")

    def setup_optimizers_before_training(self):
        """Setup optimizers and schedulers before training."""
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get("lr", 1e-4),
            betas=self.config.get("betas", (0.9, 0.995)),
            weight_decay=self.config.get("weight_decay", 0.01),
        )

        # Create learning rate scheduler
        self.scheduler = transformers.get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.get("num_warmup_steps", 100),
            num_training_steps=self.config.get("max_epochs", 10)
            * len(self.train_dataloader),
            num_cycles=self.config.get("num_cycles", 0.5),
            last_epoch=self.cur_epoch,
        )

    def extract_loss(self, output, validation=False) -> torch.Tensor:
        """Extract loss from the model output."""
        if hasattr(output, "losses"):
            losses = output.losses
        elif isinstance(output, torch.Tensor):
            losses = {"total": output}
        else:
            losses = output

        total_loss = 0
        for key in losses:
            if losses[key] is not None:
                total_loss += losses[key]
                loss_reduce = losses[key].detach()

                if validation:
                    mode = "validation"
                else:
                    mode = "train"

                self.log(
                    f"{mode}/losses/{key}",
                    loss_reduce.item(),
                )

        return total_loss

    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        # Setup optimizers before training
        self.setup_optimizers_before_training()

        # Get references to model, optimizer, and scheduler
        model, optimizer, scheduler = self.model, self.optimizer, self.scheduler

        # Create progress bar for training
        pbar_train_dataloader = tqdm(
            self.train_dataloader,
            total=len(self.train_dataloader),
            desc=f"Training (Rank {self.distributed_config.rank})",
        )

        # Training loop
        while True:  # Continue until max epoch or patience is reached
            self.cur_epoch += 1

            # Reset progress bar for new epoch
            pbar_train_dataloader.reset()
            pbar_train_dataloader.set_description(
                f"Epoch {self.cur_epoch} (Rank {self.distributed_config.rank})"
            )

            # Set sampler epoch for distributed training
            if hasattr(self.train_dataloader, "sampler") and hasattr(
                self.train_dataloader.sampler, "set_epoch"
            ):
                self.train_dataloader.sampler.set_epoch(self.cur_epoch)

            # Train for one epoch
            for data in pbar_train_dataloader:
                self.cur_step += 1

                # Check exit conditions
                if self.cur_epoch >= self.config.max_epochs or self.exit_by_patience:
                    print(
                        f"Exit requirement reached, exiting (Rank {self.distributed_config.rank})"
                    )

                    # Save final checkpoint on main process
                    if is_main_process(self.distributed_config):
                        self.save_checkpoint(for_last=True)

                    # Clean up distributed environment
                    cleanup_distributed()

                    return self.get_training_results()

                # Forward and backward pass
                model.train()

                # Move data to device
                data = self.move_to_device(data)

                # Zero gradients
                optimizer.zero_grad(set_to_none=True)

                # Forward pass with mixed precision
                with torch.autocast(
                    device_type="cuda" if torch.cuda.is_available() else "cpu",
                    dtype=torch.float16
                    if self.config.mixed_precision
                    else torch.float32,
                ):
                    # Run model
                    if isinstance(data, dict):
                        output = model(**data)
                    elif isinstance(data, list):
                        output = model(*data)
                    else:
                        output = model(data)

                    # Calculate loss
                    total_loss = self.extract_loss(output)

                # Backward pass
                total_loss.backward()

                # Update parameters
                optimizer.step()

                # Update learning rate
                scheduler.step()

                # Log learning rate and loss
                if (
                    self.cur_step % self.config.train_log_interval == 0
                    and is_main_process(self.distributed_config)
                ):
                    for index_group in range(len(optimizer.param_groups)):
                        lr = optimizer.param_groups[index_group]["lr"]
                        self.log(f"train/lr_group_{index_group}", lr)

                    self.log("train/loss", total_loss.item())

                    # Print to console
                    pbar_train_dataloader.set_postfix(
                        {"loss": f"{total_loss.item():.4f}", "lr": f"{lr:.2e}"}
                    )

            # Run validation after each epoch
            self.on_validate_start()

            # Clean up memory
            torch.cuda.empty_cache()
            gc.collect()

        # This should never be reached
        return self.get_training_results()

    def on_validate_start(self):
        """Run validation and handle checkpointing."""
        # Run validation
        metrics_dict = self.validate()

        # Handle checkpointing based on validation metrics
        if is_main_process(self.distributed_config):
            self.handle_checkpoint_with_patience(metrics_dict, set_patience=True)
            self.setup_for_patience_callback()

        # Clean up memory
        torch.cuda.empty_cache()

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model on the validation set."""
        # Switch model to evaluation mode
        self.model.eval()

        print(f"Evaluating (Rank {self.distributed_config.rank})")

        # Run evaluation
        metrics = self.evaluate()

        # Log metrics
        if metrics is not None and is_main_process(self.distributed_config):
            history_obj = {}

            for key in metrics:
                log_key = f"val/metrics/{key}"
                self.log(log_key, metrics[key])
                history_obj[key] = round(metrics[key], 4)
                print(f"{key}: {metrics[key]}")

            # Add to history
            self.history_metrics.append(
                {**history_obj, "epoch": self.cur_epoch, "step": self.cur_step}
            )

        return metrics

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model using metrics."""
        # Load metrics
        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")
        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")

        list_metrics = [accuracy_metric, f1_metric, precision_metric, recall_metric]

        # Collect predictions and references
        predictions_list = []
        references_list = []

        # Get device
        device = next(self.model.parameters()).device

        # Evaluate on validation set
        for batch in tqdm(
            self.val_dataloader,
            total=len(self.val_dataloader),
            desc=f"Evaluate (Rank {self.distributed_config.rank})",
        ):
            # Move batch to device
            batch = self.move_to_device(batch)

            # Get predictions
            predictions = self.model(**batch, return_loss=False)
            predictions_list.append(torch.argmax(predictions, dim=1).detach().cpu())
            references_list.append(batch["label_id"].detach().cpu())

        # Gather predictions and references from all processes in distributed training
        if self.distributed_config.mode != DistributedMode.NONE:
            # Convert lists to tensors
            predictions_tensor = torch.cat(predictions_list)
            references_tensor = torch.cat(references_list)

            # Create gather list
            world_size = self.distributed_config.world_size
            gather_pred = [
                torch.zeros_like(predictions_tensor) for _ in range(world_size)
            ]
            gather_ref = [
                torch.zeros_like(references_tensor) for _ in range(world_size)
            ]

            # Gather data from all processes
            torch.distributed.all_gather(gather_pred, predictions_tensor)
            torch.distributed.all_gather(gather_ref, references_tensor)

            # Only process on rank 0
            if self.distributed_config.rank == 0:
                predictions_list = gather_pred
                references_list = gather_ref
            else:
                return {}  # Non-main processes return empty metrics

        # Calculate metrics (only on main process)
        if not is_main_process(self.distributed_config):
            return {}

        # Concatenate all predictions and references
        all_predictions = torch.cat(predictions_list)
        all_references = torch.cat(references_list)

        # Calculate metrics
        results_dict = {}
        for metric in list_metrics:
            if metric.name == "accuracy":
                result_dict = metric.compute(
                    predictions=all_predictions,
                    references=all_references,
                )
            else:
                result_dict = metric.compute(
                    predictions=all_predictions,
                    references=all_references,
                    average="macro",
                )
            results_dict.update(result_dict)

        return results_dict

    def move_to_device(self, obj):
        """Move object to the device."""
        device = next(self.model.parameters()).device

        if isinstance(obj, dict):
            return {k: self.move_to_device(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.move_to_device(v) for v in obj]
        elif torch.is_tensor(obj):
            return obj.to(device)
        else:
            return obj

    def handle_checkpoint_with_patience(
        self, metrics: Dict[str, Any], set_patience=True
    ):
        """Handle checkpoint saving and early stopping based on metrics."""
        if not is_main_process(self.distributed_config):
            return

        reset_patience_flag = False

        # Initialize best metrics on first run
        if len(self.best_metrics_values) == 0:
            self.best_metrics_values = metrics
            return

        # Check if any metric is better than previous best
        for key in self.best_metrics_values:
            short_key = key.split("/")[-1]
            if metrics.get(short_key, -100) > self.best_metrics_values[key]:
                self.best_metrics_values[key] = metrics[short_key]
                self.save_checkpoint(name=f"best_{short_key}")
                reset_patience_flag = True

        # Update patience
        if set_patience:
            if reset_patience_flag:
                self.current_patience = 0
            else:
                self.current_patience += 1

    def setup_for_patience_callback(self):
        """Check if early stopping should be triggered based on patience."""
        if self.current_patience > self.max_patience:
            print("Early stopping triggered")
            self.exit_by_patience = True

    def save_checkpoint(self, name="last", for_last=False):
        """Save a checkpoint of the model."""
        if not is_main_process(self.distributed_config):
            return

        # Get the checkpoint path
        check_point_file_path = (
            f"{self.checkpoint_prefix}/{name}.pt"
            if not for_last
            else f"{self.checkpoint_prefix}/last.pt"
        )

        # Get the model state dict
        if self.distributed_config.mode != DistributedMode.NONE:
            # Use specialized save for distributed models
            save_distributed_model(
                self.model, self.distributed_config, check_point_file_path
            )
        else:
            # Save the state dict directly
            model_state_dict = self.model.state_dict()

            # Create save object
            save_obj = {
                "model": model_state_dict,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "config": self.config.to_dict()
                if hasattr(self.config, "to_dict")
                else self.config,
                "epoch": self.cur_epoch,
                "history": self.history_metrics,
                "best_metrics": self.best_metrics_values,
                "patience": self.current_patience,
            }

            # Save checkpoint
            torch.save(save_obj, check_point_file_path)

        # Clean up memory
        torch.cuda.empty_cache()

    def log(self, name: str, value: Union[torch.Tensor, float, int]):
        """Log a value to the logger."""
        if self._logger is not None and is_main_process(self.distributed_config):
            self._logger.log({name: value, "epoch": self.cur_epoch}, step=self.cur_step)

    def get_training_results(self) -> Dict[str, Any]:
        """Get the results of training."""
        if not is_main_process(self.distributed_config):
            return {}

        # Create history dataframe
        history_metrics = pd.DataFrame(self.history_metrics).round(4)

        # Return results
        return {
            "history": history_metrics,
            "best_metrics": self.best_metrics_values,
            "patience": self.current_patience,
        }


# Create a factory method to get a trainer based on configuration
def get_trainer(
    config: Union[TrainingConfig, DictConfig],
    model: torch.nn.Module,
    train_dataset: Any,
    val_dataset: Any,
    tokenizer: Optional[Any] = None,
    labels: Optional[List[str]] = None,
    logger: Optional[Any] = None,
) -> DistributedTrainer:
    """
    Factory function to create a trainer based on the configuration.

    Args:
        config: Training configuration
        model: The model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        tokenizer: Tokenizer for the model
        labels: List of class labels
        logger: Logger for tracking metrics

    Returns:
        A trainer instance
    """
    return DistributedTrainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        labels=labels,
        logger=logger,
    )
