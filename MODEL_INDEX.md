# Model Index

This file is the single source of truth for models in this repository.
Use it to quickly identify which model to run and where the relevant code lives.

## At a Glance

| Model Family | Primary Task | Main Folder | Status |
|---|---|---|---|
| DeBERTa Ranker (K-fold) | Track A pair ranking | `./` and `DBERTA/` | Active |
| DeBERTa Embedding Export | Track B embedding output | `./` and `DBERTA/` | Active |
| BGE Large Baseline/Fine-tuning | Track A baseline + triplet fine-tune | `bge_large/` | Active |
| Cross-Encoder (MiniLM) | Track A pair scoring | `bge_large/crossencoder/` | Active |
| E5 Large v2 | Track A train + eval + predict | `bge_large/e5_large/` | Active |
| SBERT / TF-IDF + SBERT baselines | Track A prototype baselines | `Approach-1_Track-A/` | Prototype |
| Track B SBERT baseline/fine-tune | Track B embeddings | `Approach-1_Track-B/` | Prototype |
| DeBERTa Siamese notebook script | Track A ranking (single-file prototype) | `DBERTA_SIAMESE/` | Prototype |

## Canonical Entrypoints

Use these first unless you explicitly need an experiment folder.

### 1) DeBERTa Ranker (Track A)
- Train: `python train.py --config configs/config.yaml`
- Predict: `python predict.py --config configs/config.yaml --test_file data/test.jsonl --output_file predictions.jsonl`
- Code: `train.py`, `predict.py`, `models/deberta_ranker.py`, `utils/dataset.py`
- Checkpoints: `checkpoints/`

### 2) DeBERTa Embeddings (Track B)
- Generate embeddings: `python generate_embeddings.py --config configs/config.yaml --test_file data/testb.jsonl --output_file submission_trackb.jsonl`
- Code: `generate_embeddings.py`
- Checkpoints: `checkpoints/`

### 3) BGE Large (Track A)
- Baseline eval: `python bge_large/evaluation_files/bge_baseline_eval.py --train_file bge_large/data/train.jsonl --dev_file bge_large/data/dev.jsonl --output_dir bge_large/outputs/bge_baseline`
- Fine-tune: `python bge_large/train.py`
- Predict: `python bge_large/evaluation_files/bge_predict.py --test_file data/test.jsonl --output_file bge_predictions.jsonl`
- Code: `bge_large/train.py`, `bge_large/evaluation_files/bge_predict.py`, `bge_large/evaluation_files/`
- Checkpoints: `bge_large/checkpoints/`

### 4) Cross-Encoder (Track A)
- Train: `python bge_large/crossencoder/train.py`
- Predict dev: `python bge_large/crossencoder/predict.py`
- Predict test: `python bge_large/crossencoder/predict_test.py`
- Checkpoints: `bge_large/crossencoder/checkpoints/`

### 5) E5 Large v2 (Track A)
- Train: `python bge_large/e5_large/train.py`
- Evaluate: `python bge_large/e5_large/evaluate.py --train_file bge_large/data/final_train.jsonl --dev_file bge_large/data/dev.jsonl --model_name_or_path bge_large/e5_large/checkpoints --output_dir bge_large/e5_large/outputs`
- Predict: `python bge_large/e5_large/predict.py --test_file data/test.jsonl --model_name_or_path bge_large/e5_large/checkpoints --output_file bge_large/e5_large/outputs/e5_predictions.jsonl`
- Code: `bge_large/e5_large/train.py`, `bge_large/e5_large/evaluate.py`, `bge_large/e5_large/predict.py`, `bge_large/e5_large/config.yaml`
- Checkpoints: `bge_large/e5_large/checkpoints/`

## Experimental and Legacy Scripts

These scripts are useful for quick experiments but are not fully productionized.

- `Approach-1_Track-A/tracka_sbert_baseline.py`
- `Approach-1_Track-A/combined_tfidf_sbert_binary.py`
- `Approach-1_Track-B/track_b_baseline.py`
- `Approach-1_Track-B/track_b_finetuning.py`
- `DBERTA_SIAMESE/Dberta_Siamese.py`

## Which Model Should I Run?

- Need the main Track A pipeline: use DeBERTa K-fold (`train.py`, `predict.py`).
- Need Track B embeddings in submission format: use `generate_embeddings.py`.
- Need a strong embedding baseline quickly: use BGE baseline eval in `bge_large/evaluation_files/`.
- Need pairwise re-ranking behavior: use the cross-encoder.
- Need fast prototype baselines for comparison: use `Approach-1_Track-A/` and `Approach-1_Track-B/`.

## Notes on Duplicate Folders

`DBERTA/` mirrors the root DeBERTa pipeline structure with similar scripts. Prefer root entrypoints unless you intentionally want the isolated `DBERTA/` copy.
