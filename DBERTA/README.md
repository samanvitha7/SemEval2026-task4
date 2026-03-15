# DBERTA Pipeline (DeBERTa v3)

This folder contains a DeBERTa-based pipeline for SemEval Task 4.
It mirrors the root-level DeBERTa workflow with local configs and scripts.

## Purpose

- Track A: pairwise prediction (`text_a_is_closer`)
- Track B: embedding extraction for submission format

## Core Files

- `train.py`: K-fold training pipeline
- `predict.py`: Track A prediction script
- `generate_embeddings.py`: Track B embedding export script
- `configs/config.yaml`: hyperparameters and paths
- `models/deberta_ranker.py`: model definition
- `utils/dataset.py`: dataset/tokenization logic

## Quick Commands

```bash
# Train
python DBERTA/train.py --config DBERTA/configs/config.yaml

# Predict Track A
python DBERTA/predict.py --config DBERTA/configs/config.yaml --test_file data/test.jsonl --output_file predictions.jsonl

# Generate Track B embeddings
python DBERTA/generate_embeddings.py --config DBERTA/configs/config.yaml --test_file data/testb.jsonl --output_file submission_trackb.jsonl
```

## Artifacts

- Checkpoints: `DBERTA/checkpoints/`
- SLURM scripts: `DBERTA/HPC_sh/`

## Notes

- Root-level scripts provide the same DeBERTa flow and are usually the primary entrypoint.
- Use this folder when you want an isolated DeBERTa workflow.
