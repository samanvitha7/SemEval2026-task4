import random
import torch
from torch.utils.data import Dataset


class SemEvalDataset(Dataset):
    def __init__(self, df, tokenizer, max_len: int, training: bool = True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.training = training

    def __len__(self):
        return len(self.df)

    def _tokenize(self, text):
        return self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        anchor = str(row["anchor_text"])
        text_a = str(row["text_a"])
        text_b = str(row["text_b"])

        # Handle test data without labels
        if "text_a_is_closer" in row:
            label = 1 if row["text_a_is_closer"] else 0
        else:
            label = 0  # Dummy label for test data

        if self.training and random.random() < 0.5:
            text_a, text_b = text_b, text_a
            label = 1 - label

        anc = self._tokenize(anchor)
        a = self._tokenize(text_a)
        b = self._tokenize(text_b)

        return {
            "anc_ids": anc["input_ids"].squeeze(0),
            "anc_mask": anc["attention_mask"].squeeze(0),
            "a_ids": a["input_ids"].squeeze(0),
            "a_mask": a["attention_mask"].squeeze(0),
            "b_ids": b["input_ids"].squeeze(0),
            "b_mask": b["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }
