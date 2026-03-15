# E5 Large v2 (Track A)

## Purpose
- Task/Track: Track A (pairwise ranking with anchor, text_a, text_b)
- Goal: Provide an E5-based model with the same train/predict/evaluate workflow as other model families
- Status: Active experimental pipeline

## Core Files
- `train.py`: Triplet-loss fine-tuning for `intfloat/e5-large-v2`
- `predict.py`: Track A inference on test JSONL, outputs submission-ready JSONL
- `evaluate.py`: Train/dev evaluation with accuracy + detailed CSV outputs
- `config.yaml`: Local defaults for model, paths, and batch sizes

## Quick Commands

Run from repository root:

```bash
# 1) Train
python bge_large/e5_large/train.py \
	--train_file bge_large/data/final_train.jsonl \
	--output_dir bge_large/e5_large/checkpoints

# 2) Evaluate
python bge_large/e5_large/evaluate.py \
	--train_file bge_large/data/final_train.jsonl \
	--dev_file bge_large/data/dev.jsonl \
	--model_name_or_path bge_large/e5_large/checkpoints \
	--output_dir bge_large/e5_large/outputs

# 3) Predict
python bge_large/e5_large/predict.py \
	--test_file data/test.jsonl \
	--model_name_or_path bge_large/e5_large/checkpoints \
	--output_file bge_large/e5_large/outputs/e5_predictions.jsonl
```

## HPC (SLURM) Commands

From repository root:

```bash
# Train
sbatch bge_large/HPC_sh/run_e5_train.sh

# Evaluate (train + dev)
sbatch bge_large/HPC_sh/run_e5_eval.sh

# Predict (test)
sbatch bge_large/HPC_sh/run_e5_predict.sh
```

Optional positional arguments are supported for custom paths:

```bash
sbatch bge_large/HPC_sh/run_e5_train.sh data/final_train.jsonl e5_large/checkpoints
sbatch bge_large/HPC_sh/run_e5_eval.sh data/final_train.jsonl data/dev.jsonl e5_large/checkpoints e5_large/outputs
sbatch bge_large/HPC_sh/run_e5_predict.sh data/test.jsonl e5_large/checkpoints e5_large/outputs/e5_predictions.jsonl
```

## Data Contract
- Input JSONL fields for Track A: `anchor_text`, `text_a`, `text_b`
- Label field for training/eval: `text_a_is_closer` (0 or 1)
- Prediction output format: one JSON object per line with `{"text_a_is_closer": bool}`

## Artifacts
- Trained model/checkpoints: `bge_large/e5_large/checkpoints/`
- Evaluation CSVs: `bge_large/e5_large/outputs/`
- Prediction file: `bge_large/e5_large/outputs/e5_predictions.jsonl`
