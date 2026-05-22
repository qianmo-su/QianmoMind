import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be positive")

        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

        self.lora_a = nn.Linear(base_layer.in_features, r, bias=False)
        self.lora_b = nn.Linear(r, base_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)
        self.lora_a.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)
        self.lora_b.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)

        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.base_layer(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


def freeze_model(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def mark_lora_trainable(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name


def inject_lora(model: nn.Module, target_modules, r: int = 8, alpha: int = 16, dropout: float = 0.0) -> None:
    target_modules = set(target_modules)

    for module in model.modules():
        for child_name, child in list(module.named_children()):
            if child_name in target_modules and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))

    mark_lora_trainable(model)


def lora_state_dict(model: nn.Module):
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if ".lora_" in name
    }


def load_lora_state_dict(model: nn.Module, state_dict) -> None:
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    unexpected_lora = [name for name in unexpected if ".lora_" in name]
    if unexpected_lora:
        raise RuntimeError(f"unexpected LoRA keys: {unexpected_lora}")
