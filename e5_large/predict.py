#!/usr/bin/env python3
"""Generate Track A predictions using E5-Large-v2."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Predict Track A labels with E5")
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Path to test JSONL containing anchor_text/text_a/text_b",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=str(root / "bge_large" / "e5_large" / "outputs" / "e5_predictions.jsonl"),
        help="Path to output prediction JSONL",
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


def predict(df: pd.DataFrame, model: SentenceTransformer, batch_size: int) -> List[bool]:
    required = {"anchor_text", "text_a", "text_b"}
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
    return (sim_a > sim_b).tolist()


def main() -> None:
    args = parse_args()

    test_file = Path(args.test_file).resolve()
    output_file = Path(args.output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading model: {args.model_name_or_path}")
    model = SentenceTransformer(args.model_name_or_path, device=device)
    model.eval()

    test_df = load_jsonl_safe(test_file)
    predictions = predict(test_df, model, batch_size=args.batch_size)

    with output_file.open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps({"text_a_is_closer": bool(pred)}) + "\n")

    print(f"Saved predictions to: {output_file}")
    print(f"text_a_is_closer=True for {sum(predictions)}/{len(predictions)} rows")


if __name__ == "__main__":
    main()
