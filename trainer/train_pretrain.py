import argparse
import math
import os
import sys
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dataset.lm_dataset import PretrainDataset
from model.model import MokioMindConfig, MokioMindForCausalLM
from trainer.trainer_utils import (
    build_cosine_scheduler,
    get_amp_dtype,
    get_device,
    save_checkpoint,
    set_seed,
)


@dataclass
class TokenizerOutput:
    input_ids: list[int]


class TinyCharTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self):
        chars = (
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            " .,;:!?()[]{}+-=*/_\"'\n"
        )
        self.char_to_id = {ch: idx + 3 for idx, ch in enumerate(chars)}
        self.unk_token_id = len(self.char_to_id) + 3

    def __len__(self):
        return self.unk_token_id + 1

    def __call__(
        self,
        text,
        add_special_tokens=False,
        max_length=None,
        truncation=False,
    ):
        ids = [self.char_to_id.get(ch, self.unk_token_id) for ch in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return TokenizerOutput(input_ids=ids)


class SmokeDataset(Dataset):
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = [
            "hello world. this is a tiny language model training sample.",
            "the model learns to predict the next token from previous tokens.",
            "small tests should run quickly before using real pretrain data.",
            "qianmo mind smoke test checks forward backward and optimizer step.",
        ] * 16

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        token_ids = self.tokenizer(
            self.samples[idx],
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True,
        ).input_ids
        token_ids = [self.tokenizer.bos_token_id] + token_ids + [self.tokenizer.eos_token_id]
        input_ids = token_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(token_ids))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "labels": labels}


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Minimal MokioMind pretrain runner.")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="checkpoints/pretrain_smoke")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_kv_heads", type=int, default=2)
    return parser


def build_tokenizer_and_dataset(args):
    if args.data_path and args.tokenizer_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        dataset = PretrainDataset(args.data_path, tokenizer, max_length=args.max_length)
        return tokenizer, dataset, False

    tokenizer = TinyCharTokenizer()
    dataset = SmokeDataset(tokenizer, max_length=args.max_length)
    return tokenizer, dataset, True


def train():
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    tokenizer, dataset, is_smoke = build_tokenizer_and_dataset(args)
    device = get_device()
    amp_dtype = get_amp_dtype(device)

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
    model = MokioMindForCausalLM(config).to(device)
    model.train()

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
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

    print(f"mode={'smoke' if is_smoke else 'pretrain'} device={device} dtype={amp_dtype}")
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
