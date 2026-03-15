# SemEval 2026 Task 4 - Narrative Similarity and Representation Learning

This repository contains model training, inference, and evaluation code used by the SVNIT Surat team for SemEval 2026 Task 4.

## Quick Navigation

- Model index (single source of truth): `MODEL_INDEX.md`
- Main Track A pipeline (DeBERTa ranker): `train.py`, `predict.py`
- Main Track B pipeline (embedding export): `generate_embeddings.py`
- BGE family experiments and evaluation: `bge_large/`
- Early prototype baselines: `Approach-1_Track-A/`, `Approach-1_Track-B/`

## Project Areas

- `./` (root): canonical DeBERTa K-fold training and inference scripts
- `DBERTA/`: mirrored DeBERTa pipeline in isolated folder form
- `bge_large/`: BGE baseline/fine-tuning, cross-encoder, E5-large experiments, and evaluation scripts
- `Approach-1_Track-A/`: SBERT and TF-IDF+SBERT prototype baselines for Track A
- `Approach-1_Track-B/`: sentence-transformer Track B prototype scripts
- `DBERTA_SIAMESE/`: notebook-exported DeBERTa Siamese prototype

## Quick Start

### 1) Train DeBERTa for Track A

```bash
python train.py --config configs/config.yaml
```

### 2) Predict Track A labels

```bash
python predict.py --config configs/config.yaml --test_file data/test.jsonl --output_file predictions.jsonl
```

### 3) Generate Track B embeddings

```bash
python generate_embeddings.py --config configs/config.yaml --test_file data/testb.jsonl --output_file submission_trackb.jsonl
```

### 4) Run BGE baseline (Track A)

```bash
python bge_large/evaluation_files/bge_baseline_eval.py --train_file bge_large/data/train.jsonl --dev_file bge_large/data/dev.jsonl --output_dir bge_large/outputs/bge_baseline
```

## Which Model Should I Use?

- Need maintained Track A training/prediction: use root DeBERTa pipeline.
- Need maintained Track B embeddings: use root `generate_embeddings.py`.
- Need strong embedding baseline quickly: use BGE baseline scripts in `bge_large/evaluation_files/`.
- Need prototype or ablation comparisons: check `Approach-1_Track-A/`, `Approach-1_Track-B/`, and `DBERTA_SIAMESE/`.

## Documentation Standard

For consistent documentation in model folders, use template:

- `docs/MODEL_README_TEMPLATE.md`
