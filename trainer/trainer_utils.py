import os
import random
from pathlib import Path

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_amp_dtype(device: torch.device):
    if device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def save_checkpoint(
    output_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config,
) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(output_dir, "last.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": config.to_dict() if hasattr(config, "to_dict") else None,
        },
        path,
    )
    return path
