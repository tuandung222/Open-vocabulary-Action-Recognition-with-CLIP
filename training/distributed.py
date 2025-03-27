import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    default_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Callable
from enum import Enum, auto


class DistributedMode(Enum):
    """Enum for distributed training modes."""

    NONE = auto()  # Single GPU or CPU training
    DDP = auto()  # DistributedDataParallel
    FSDP = auto()  # FullyShardedDataParallel


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""

    mode: DistributedMode = DistributedMode.NONE
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: str = "nccl"
    master_addr: str = "localhost"
    master_port: str = "12355"
    mixed_precision: bool = True
    cpu_offload: bool = False

    # FSDP specific configurations
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    backward_prefetch: Optional[BackwardPrefetch] = BackwardPrefetch.BACKWARD_PRE
    min_num_params: int = 1e6  # Minimum number of parameters for a layer to be wrapped
    ignored_modules: Optional[Tuple[torch.nn.Module]] = None


def setup_distributed_environment(config: DistributedConfig) -> DistributedConfig:
    """
    Set up the distributed training environment.

    Args:
        config: The distributed configuration.

    Returns:
        Updated configuration with proper rank and world_size.
    """
    if config.mode == DistributedMode.NONE:
        return config

    # Set environment variables for distributed training
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = config.master_port

    # Initialize the process group
    if "WORLD_SIZE" in os.environ:
        config.world_size = int(os.environ["WORLD_SIZE"])
    if "RANK" in os.environ:
        config.rank = int(os.environ["RANK"])
    if "LOCAL_RANK" in os.environ:
        config.local_rank = int(os.environ["LOCAL_RANK"])

    # Initialize the distributed backend
    dist.init_process_group(
        backend=config.backend,
        world_size=config.world_size,
        rank=config.rank,
    )

    print(
        f"Initialized process group: rank={config.rank}, world_size={config.world_size}"
    )

    # Set the device for this process
    torch.cuda.set_device(config.local_rank)

    return config


def cleanup_distributed():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_mixed_precision_policy(enabled: bool = True) -> Optional[MixedPrecision]:
    """
    Get mixed precision policy for FSDP.

    Args:
        enabled: Whether to enable mixed precision.

    Returns:
        MixedPrecision policy or None if not enabled.
    """
    if not enabled:
        return None

    return MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )


def wrap_model_for_distributed(
    model: nn.Module, config: DistributedConfig
) -> nn.Module:
    """
    Wrap a model for distributed training based on the config.

    Args:
        model: The model to wrap.
        config: The distributed configuration.

    Returns:
        The wrapped model.
    """
    # Single GPU or CPU mode
    if config.mode == DistributedMode.NONE:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        return model

    # Move model to the correct device
    device = torch.device(f"cuda:{config.local_rank}")
    model = model.to(device)

    # DDP mode
    if config.mode == DistributedMode.DDP:
        model = DDP(
            model,
            device_ids=[config.local_rank],
            output_device=config.local_rank,
            find_unused_parameters=False,
        )
        return model

    # FSDP mode
    if config.mode == DistributedMode.FSDP:
        # Define the wrapping policy based on the configuration
        def auto_wrap_policy(module, recurse, unwrapped_params, module_is_root):
            return default_auto_wrap_policy(
                module,
                recurse,
                unwrapped_params,
                module_is_root=module_is_root,
                min_num_params=config.min_num_params,
                ignored_modules=config.ignored_modules,
            )

        # Get mixed precision policy
        mixed_precision_policy = get_mixed_precision_policy(config.mixed_precision)

        # Configure CPU offload if enabled
        cpu_offload = CPUOffload(offload_params=True) if config.cpu_offload else None

        # Wrap the model with FSDP
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision_policy,
            sharding_strategy=config.sharding_strategy,
            cpu_offload=cpu_offload,
            backward_prefetch=config.backward_prefetch,
            device_id=config.local_rank,
        )

        return model

    raise ValueError(f"Unsupported distributed mode: {config.mode}")


def save_distributed_model(
    model: nn.Module,
    config: DistributedConfig,
    save_path: str,
):
    """
    Save a distributed model.

    Args:
        model: The distributed model to save.
        config: The distributed configuration.
        save_path: Path to save the model.
    """
    if config.mode == DistributedMode.NONE or config.mode == DistributedMode.DDP:
        # For single GPU/CPU or DDP, only save on the main process
        if config.rank == 0 or config.mode == DistributedMode.NONE:
            if isinstance(model, DDP):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
    elif config.mode == DistributedMode.FSDP:
        # For FSDP, use the FSDP state dict utilities
        full_state_dict_config = FullStateDictConfig(
            offload_to_cpu=True,
            rank0_only=True,
        )

        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, full_state_dict_config
        ):
            state_dict = model.state_dict()

            if config.rank == 0:
                torch.save(state_dict, save_path)

    # Synchronize all processes
    if config.mode != DistributedMode.NONE:
        dist.barrier()


def load_distributed_model(
    model: nn.Module,
    config: DistributedConfig,
    load_path: str,
) -> nn.Module:
    """
    Load a saved model for distributed training.

    Args:
        model: The model architecture (not wrapped).
        config: The distributed configuration.
        load_path: Path to load the model from.

    Returns:
        The loaded model (not wrapped for distributed training).
    """
    # Load the state dict
    state_dict = torch.load(load_path, map_location="cpu")

    # Apply the state dict to the model
    if isinstance(model, (DDP, FSDP)):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    return model


def get_distributed_sampler(
    dataset,
    config: DistributedConfig,
    shuffle: bool = True,
):
    """
    Get a distributed sampler for the dataset.

    Args:
        dataset: The dataset.
        config: The distributed configuration.
        shuffle: Whether to shuffle the dataset.

    Returns:
        A distributed sampler or None if not in distributed mode.
    """
    if config.mode == DistributedMode.NONE:
        return None

    return torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=config.world_size,
        rank=config.rank,
        shuffle=shuffle,
    )


def is_main_process(config: DistributedConfig) -> bool:
    """
    Check if the current process is the main process.

    Args:
        config: The distributed configuration.

    Returns:
        True if the current process is the main process, False otherwise.
    """
    return config.rank == 0 or config.mode == DistributedMode.NONE
