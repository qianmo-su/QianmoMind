import os
import random
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR


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
    scheduler=None,
) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(output_dir, "last.pt")
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": config.to_dict() if hasattr(config, "to_dict") else None,
    }
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    torch.save(checkpoint, path)
    return path


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    max_steps: int,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1]")

    warmup_steps = min(warmup_steps, max_steps)

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)

        decay_steps = max(max_steps - warmup_steps, 1)
        progress = float(current_step - warmup_steps) / float(decay_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)
