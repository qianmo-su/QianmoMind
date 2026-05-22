import argparse
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from model.lora import inject_lora, load_lora_state_dict
from model.model import MokioMindConfig, MokioMindForCausalLM


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Chat with a trained MokioMind checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pretrain/last.pt")
    parser.add_argument("--lora_adapter", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--raw_prompt", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    return parser


def apply_chat_template(prompt):
    return f"用户：{prompt}\n助手："


def load_model(checkpoint_path, tokenizer, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config_dict = checkpoint.get("config")
    if config_dict is None:
        raise ValueError("Checkpoint has no config. Please use a checkpoint saved by trainer/train_pretrain.py.")

    config = MokioMindConfig(**config_dict)
    config.vocab_size = len(tokenizer)
    model = MokioMindForCausalLM(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    return model


def load_lora_adapter(model, adapter_path, device):
    checkpoint = torch.load(adapter_path, map_location="cpu")
    lora_config = checkpoint["lora_config"]
    inject_lora(
        model,
        target_modules=lora_config["target_modules"],
        r=lora_config["r"],
        alpha=lora_config["alpha"],
        dropout=0.0,
    )
    load_lora_state_dict(model, checkpoint["adapter"])
    model.to(device)
    model.eval()


def apply_repetition_penalty(logits, generated_ids, penalty):
    if penalty == 1.0:
        return logits
    for token_id in set(generated_ids):
        if logits[token_id] < 0:
            logits[token_id] *= penalty
        else:
            logits[token_id] /= penalty
    return logits


def filter_logits(logits, top_k=0, top_p=1.0):
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        logits[logits < values[..., -1, None]] = -float("inf")

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float("inf")

    return logits


def apply_no_repeat_ngram(logits, generated_ids, ngram_size):
    if ngram_size <= 0 or len(generated_ids) + 1 < ngram_size:
        return logits

    prefix = tuple(generated_ids[-(ngram_size - 1):])
    banned_tokens = []

    for i in range(len(generated_ids) - ngram_size + 1):
        ngram = tuple(generated_ids[i: i + ngram_size])
        if ngram[:-1] == prefix:
            banned_tokens.append(ngram[-1])

    if banned_tokens:
        logits[banned_tokens] = -float("inf")

    return logits


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    no_repeat_ngram_size,
):
    input_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    if bos_id is not None:
        input_ids = [bos_id] + input_ids

    generated = list(input_ids)
    max_context = model.config.max_position_embeddings

    for _ in range(max_new_tokens):
        context_ids = generated[-max_context:]
        x = torch.tensor([context_ids], dtype=torch.long, device=device)
        outputs = model(input_ids=x)
        logits = outputs["logits"][0, -1, :].float()
        logits = apply_repetition_penalty(logits, generated, repetition_penalty)
        logits = apply_no_repeat_ngram(logits, generated, no_repeat_ngram_size)

        if temperature <= 0:
            next_id = torch.argmax(logits).item()
        else:
            logits = logits / temperature
            logits = filter_logits(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()

        generated.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break

    new_tokens = generated[len(input_ids):]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    args = build_arg_parser().parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args.checkpoint, tokenizer, device)
    if args.lora_adapter:
        load_lora_adapter(model, args.lora_adapter, device)
    print(f"loaded checkpoint={args.checkpoint} device={device}")
    if args.lora_adapter:
        print(f"loaded lora_adapter={args.lora_adapter}")

    if args.prompt:
        prompt = args.prompt if args.raw_prompt else apply_chat_template(args.prompt)
        print(generate(
            model,
            tokenizer,
            prompt,
            device,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.top_p,
            args.repetition_penalty,
            args.no_repeat_ngram_size,
        ))
        return

    while True:
        prompt = input("\nuser> ").strip()
        if prompt.lower() in {"exit", "quit", "q"}:
            break
        if not prompt:
            continue
        model_prompt = prompt if args.raw_prompt else apply_chat_template(prompt)
        response = generate(
            model,
            tokenizer,
            model_prompt,
            device,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.top_p,
            args.repetition_penalty,
            args.no_repeat_ngram_size,
        )
        print(f"assistant> {response}")


if __name__ == "__main__":
    main()
