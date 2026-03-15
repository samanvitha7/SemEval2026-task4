#!/usr/bin/env python3
"""Train E5-Large-v2 for SemEval Track A triplet ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader


def project_root() -> Path:
    # .../SemEval2026-task4/bge_large/e5_large/train.py -> repo root
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Train E5-Large-v2 with triplet loss")
    parser.add_argument(
        "--model_name",
        type=str,
        default="intfloat/e5-large-v2",
        help="HF model name or local sentence-transformer path",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=str(root / "bge_large" / "data" / "final_train.jsonl"),
        help="Path to Track A train JSONL",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(root / "bge_large" / "e5_large" / "checkpoints"),
        help="Directory to save trained model/checkpoints",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--triplet_margin", type=float, default=0.35)
    return parser.parse_args()


def load_jsonl_safe(path: Path) -> pd.DataFrame:
    rows = []
    bad_lines = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad_lines += 1
                if bad_lines <= 3:
                    print(f"Skipped malformed line {line_no} in {path.name}")

    print(f"Loaded {len(rows)} rows from {path} | Skipped {bad_lines} malformed lines")
    return pd.DataFrame(rows)


def create_training_examples(df: pd.DataFrame) -> Tuple[List[InputExample], int]:
    required = {"anchor_text", "text_a", "text_b", "text_a_is_closer"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    examples: List[InputExample] = []
    skipped = 0

    for _, row in df.iterrows():
        if pd.isna(row["anchor_text"]) or pd.isna(row["text_a"]) or pd.isna(row["text_b"]):
            skipped += 1
            continue

        anchor = str(row["anchor_text"]).strip()
        text_a = str(row["text_a"]).strip()
        text_b = str(row["text_b"]).strip()

        if not anchor or not text_a or not text_b:
            skipped += 1
            continue

        label = int(row["text_a_is_closer"])
        positive, negative = (text_a, text_b) if label == 1 else (text_b, text_a)
        examples.append(InputExample(texts=[anchor, positive, negative]))

    return examples, skipped


def main() -> None:
    args = parse_args()

    train_path = Path(args.train_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("E5-Large-v2 Training")
    print("=" * 70)
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Model: {args.model_name}")
    print(f"Train file: {train_path}")
    print(f"Output dir: {output_dir}")

    train_df = load_jsonl_safe(train_path)
    train_examples, skipped = create_training_examples(train_df)

    print(f"Training examples: {len(train_examples)}")
    if skipped:
        print(f"Skipped rows with missing text: {skipped}")

    model = SentenceTransformer(args.model_name, device=("cuda" if torch.cuda.is_available() else "cpu"))
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.TripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.COSINE,
        triplet_margin=args.triplet_margin,
    )

    warmup_steps = int(len(train_dataloader) * args.epochs * 0.05)

    print("\nTraining config")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Epochs: {args.epochs}")
    print(f"- LR: {args.learning_rate}")
    print(f"- Triplet margin: {args.triplet_margin}")
    print(f"- Warmup steps: {warmup_steps}")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        output_path=str(output_dir),
        show_progress_bar=True,
        optimizer_params={"lr": args.learning_rate},
        use_amp=torch.cuda.is_available(),
        checkpoint_save_steps=max(1, len(train_dataloader)),
        checkpoint_save_total_limit=2,
    )

    print("\nTraining complete")
    print(f"Saved model/checkpoints to: {output_dir}")


if __name__ == "__main__":
    main()
