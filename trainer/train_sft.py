import argparse
import math
import os
import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dataset.sft_dataset import SFTDataset
from model.model import MokioMindConfig, MokioMindForCausalLM
from trainer.trainer_utils import (
    build_cosine_scheduler,
    get_amp_dtype,
    get_device,
    save_checkpoint,
    set_seed,
)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Minimal MokioMind SFT runner.")
    parser.add_argument("--data_path", type=str, default="dataset/sft_t2t_mini.jsonl")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer")
    parser.add_argument("--init_from", type=str, default="checkpoints/pretrain/last.pt")
    parser.add_argument("--output_dir", type=str, default="checkpoints/sft")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_kv_heads", type=int, default=2)
    return parser


def build_model(args, tokenizer, device):
    if args.init_from and os.path.exists(args.init_from):
        checkpoint = torch.load(args.init_from, map_location="cpu")
        config_dict = checkpoint.get("config")
        if config_dict is None:
            raise ValueError(f"checkpoint has no config: {args.init_from}")
        config = MokioMindConfig(**config_dict)
        config.vocab_size = len(tokenizer)
        config.max_position_embeddings = args.max_length
        model = MokioMindForCausalLM(config)
        model.load_state_dict(checkpoint["model"], strict=True)
        print(f"loaded init checkpoint={args.init_from}")
    else:
        config = MokioMindConfig(
            vocab_size=len(tokenizer),
            hidden_size=args.hidden_size,
            intermediate_size=args.hidden_size * 4,
            max_position_embeddings=args.max_length,
            num_attention_heads=args.num_heads,
            num_hidden_layers=args.num_layers,
            num_key_value_heads=args.num_kv_heads,
            flash_attention=False,
        )
        model = MokioMindForCausalLM(config)
        print("init_from not found; initialized model from args")

    model.to(device)
    return model, config


def train():
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = SFTDataset(args.data_path, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = get_device()
    amp_dtype = get_amp_dtype(device)
    model, config = build_model(args, tokenizer, device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
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

    print(f"mode=sft device={device} dtype={amp_dtype}")
    print(f"dataset_size={len(dataset)} vocab_size={len(tokenizer)} max_steps={args.max_steps}")
    print(
        f"lr={args.learning_rate} warmup_steps={warmup_steps} "
        f"min_lr_ratio={args.min_lr_ratio}"
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
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
                ckpt_path = save_checkpoint(
                    args.output_dir,
                    model,
                    optimizer,
                    global_step,
                    config,
                    scheduler=scheduler,
                )
                print(f"saved={ckpt_path}")

            if global_step >= args.max_steps:
                break

    ckpt_path = save_checkpoint(
        args.output_dir,
        model,
        optimizer,
        global_step,
        config,
        scheduler=scheduler,
    )
    print(f"done step={global_step} checkpoint={ckpt_path}")


if __name__ == "__main__":
    train()
