import json
import os

import torch
from torch.utils.data import Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load_samples(data_path)

    def _load_samples(self, data_path):
        samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {data_path}:{line_no}") from exc

                conversations = []
                for message in sample.get("conversations", []):
                    conversations.append(
                        {
                            "role": message.get("role"),
                            "content": message.get("content", ""),
                        }
                    )
                samples.append({"conversations": conversations})

        return samples

    def __len__(self):
        return len(self.samples)

    def _encode(self, text):
        return self.tokenizer(
            text,
            add_special_tokens=False,
        ).input_ids

    def __getitem__(self, idx):
        sample = self.samples[idx]
        conversations = sample["conversations"]

        input_ids = []
        labels = []

        if self.tokenizer.bos_token_id is not None:
            input_ids.append(self.tokenizer.bos_token_id)
            labels.append(-100)

        for message in conversations:
            role = message.get("role")
            content = str(message.get("content", ""))

            if role == "user":
                text = f"用户：{content}\n"
                token_ids = self._encode(text)
                input_ids.extend(token_ids)
                labels.extend([-100] * len(token_ids))
            elif role == "assistant":
                prefix_ids = self._encode("助手：")
                answer_ids = self._encode(content)
                if self.tokenizer.eos_token_id is not None:
                    answer_ids = answer_ids + [self.tokenizer.eos_token_id]
                input_ids.extend(prefix_ids + answer_ids)
                labels.extend([-100] * len(prefix_ids) + answer_ids)
                newline_ids = self._encode("\n")
                input_ids.extend(newline_ids)
                labels.extend([-100] * len(newline_ids))

        input_ids = input_ids[: self.max_length]
        labels = labels[: self.max_length]

        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids.extend([self.tokenizer.pad_token_id] * pad_len)
            labels.extend([-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
