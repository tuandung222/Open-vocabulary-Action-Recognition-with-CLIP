"""
Distributed training utilities.
"""

import torch.distributed as dist

def is_main_process():
    """Check if this is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0

def get_rank():
    """Get the rank of the current process."""
    return dist.get_rank() if dist.is_initialized() else 0

def get_world_size():
    """Get the world size (number of processes)."""
    return dist.get_world_size() if dist.is_initialized() else 1