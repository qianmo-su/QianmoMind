import argparse
import math
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dataset.sft_dataset import SFTDataset
from model.lora import inject_lora, lora_state_dict
from model.model import MokioMindConfig, MokioMindForCausalLM
from trainer.trainer_utils import build_cosine_scheduler, get_amp_dtype, get_device, set_seed


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train LoRA adapters for MokioMind.")
    parser.add_argument("--data_path", type=str, default="dataset/identity_sft_example.jsonl")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer")
    parser.add_argument("--base_checkpoint", type=str, default="checkpoints/sft_full/last.pt")
    parser.add_argument("--output_dir", type=str, default="checkpoints/lora_identity")
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)
    return parser


def load_base_model(checkpoint_path, tokenizer, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config_dict = checkpoint.get("config")
    if config_dict is None:
        raise ValueError(f"checkpoint has no config: {checkpoint_path}")

    config = MokioMindConfig(**config_dict)
    config.vocab_size = len(tokenizer)
    model = MokioMindForCausalLM(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    return model, config


def save_lora(output_dir, model, config, args, step):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(output_dir, "adapter.pt")
    torch.save(
        {
            "adapter": lora_state_dict(model),
            "step": step,
            "base_config": config.to_dict(),
            "lora_config": {
                "target_modules": args.target_modules.split(","),
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
            },
        },
        path,
    )
    return path


def train():
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = get_device()
    amp_dtype = get_amp_dtype(device)
    model, config = load_base_model(args.base_checkpoint, tokenizer, device)

    target_modules = [name.strip() for name in args.target_modules.split(",") if name.strip()]
    inject_lora(
        model,
        target_modules=target_modules,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    model.to(device)
    model.train()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"mode=lora device={device} dtype={amp_dtype}")
    print(f"target_modules={target_modules} trainable={trainable_params} total={total_params}")

    dataset = SFTDataset(args.data_path, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    warmup_steps = args.warmup_steps
    if warmup_steps == 0 and args.warmup_ratio > 0:
        warmup_steps = int(args.max_steps * args.warmup_ratio)
    scheduler = build_cosine_scheduler(
        optimizer=optimizer,
        max_steps=args.max_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    global_step = 0
    running_loss = 0.0

    while global_step < args.max_steps:
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs["loss"]

            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {global_step + 1}: {loss.item()}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            running_loss += loss.item()

            if global_step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
                lr = scheduler.get_last_lr()[0]
                print(f"step={global_step:04d} loss={avg_loss:.4f} ppl={ppl:.2f} lr={lr:.2e}")
                running_loss = 0.0

            if global_step % args.save_every == 0:
                path = save_lora(args.output_dir, model, config, args, global_step)
                print(f"saved={path}")

            if global_step >= args.max_steps:
                break

    path = save_lora(args.output_dir, model, config, args, global_step)
    print(f"done step={global_step} adapter={path}")


if __name__ == "__main__":
    train()
