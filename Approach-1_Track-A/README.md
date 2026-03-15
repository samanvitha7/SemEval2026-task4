# Approach 1 - Track A Prototypes

## Purpose
- Task/Track: Track A (pairwise narrative similarity classification)
- Goal: Quick baselines using SBERT and TF-IDF+SBERT feature fusion
- Status: Prototype

## Files
- `tracka_sbert_baseline.py`: SBERT embeddings + logistic regression classifier
- `combined_tfidf_sbert_binary.py`: TF-IDF and SBERT feature concatenation + logistic regression

## Usage
These scripts were exported from notebooks and contain hardcoded local/Colab-style paths.
Update file paths before running.

```bash
python Approach-1_Track-A/tracka_sbert_baseline.py
python Approach-1_Track-A/combined_tfidf_sbert_binary.py
```

## Notes
- Best used as comparison baselines, not the main training pipeline.
- For production-style Track A training and inference, use root `train.py` and `predict.py`.
