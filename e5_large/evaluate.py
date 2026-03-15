#!/usr/bin/env python3
"""Evaluate E5-Large-v2 on Track A train/dev data."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Evaluate E5 model on Track A")
    parser.add_argument(
        "--train_file",
        type=str,
        default=str(root / "bge_large" / "data" / "final_train.jsonl"),
        help="Path to train JSONL",
    )
    parser.add_argument(
        "--dev_file",
        type=str,
        default=str(root / "bge_large" / "data" / "dev.jsonl"),
        help="Path to dev JSONL",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(root / "bge_large" / "e5_large" / "outputs"),
        help="Directory for CSV outputs",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=str(root / "bge_large" / "e5_large" / "checkpoints"),
        help="Local checkpoint dir or HF model id",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def load_jsonl_safe(path: Path) -> pd.DataFrame:
    rows = []
    bad = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
                if bad <= 3:
                    print(f"Skipped malformed line {line_no} in {path.name}")
    print(f"Loaded {len(rows)} rows from {path} | Skipped {bad} malformed lines")
    return pd.DataFrame(rows)


def clean_text(text: str) -> str:
    value = str(text).replace("\n", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", value).strip()


def score_triplets(
    model: SentenceTransformer,
    df: pd.DataFrame,
    batch_size: int,
) -> Tuple[List[int], List[float], List[float]]:
    required = {"anchor_text", "text_a", "text_b", "text_a_is_closer"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    anchor = df["anchor_text"].apply(clean_text).tolist()
    text_a = df["text_a"].apply(clean_text).tolist()
    text_b = df["text_b"].apply(clean_text).tolist()

    emb_anchor = model.encode(anchor, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    emb_a = model.encode(text_a, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    emb_b = model.encode(text_b, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)

    sim_a = np.sum(emb_anchor * emb_a, axis=1)
    sim_b = np.sum(emb_anchor * emb_b, axis=1)
    preds = (sim_a > sim_b).astype(int)

    return preds.tolist(), sim_a.tolist(), sim_b.tolist()


def evaluate_split(
    model: SentenceTransformer,
    df: pd.DataFrame,
    batch_size: int,
    split_name: str,
    output_dir: Path,
) -> float:
    preds, sim_a, sim_b = score_triplets(model, df, batch_size=batch_size)
    gold = df["text_a_is_closer"].astype(int).tolist()
    acc = float(np.mean([p == g for p, g in zip(preds, gold)]))

    out_df = df.copy()
    out_df["prediction"] = preds
    out_df["similarity_a"] = sim_a
    out_df["similarity_b"] = sim_b
    out_df["correct"] = (out_df["prediction"] == out_df["text_a_is_closer"].astype(int)).astype(int)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"{split_name}_e5_eval_{stamp}.csv"
    out_df.to_csv(out_path, index=False)

    correct = int((out_df["correct"] == 1).sum())
    total = len(out_df)
    print(f"{split_name.upper()} accuracy: {acc:.4f} ({correct}/{total})")
    print(f"Saved detailed results: {out_path}")

    return acc


def main() -> None:
    args = parse_args()

    train_file = Path(args.train_file).resolve()
    dev_file = Path(args.dev_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading model: {args.model_name_or_path}")
    model = SentenceTransformer(args.model_name_or_path, device=device)
    model.eval()

    train_df = load_jsonl_safe(train_file)
    dev_df = load_jsonl_safe(dev_file)

    print("\nEvaluating train split")
    train_acc = evaluate_split(model, train_df, args.batch_size, "train", output_dir)

    print("\nEvaluating dev split")
    dev_acc = evaluate_split(model, dev_df, args.batch_size, "dev", output_dir)

    print("\nSummary")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Dev accuracy:   {dev_acc:.4f}")


if __name__ == "__main__":
    main()
