# Approach 1 - Track B Prototypes

## Purpose
- Task/Track: Track B (embedding generation)
- Goal: Baseline and simple fine-tuning experiments with sentence-transformers
- Status: Prototype

## Files
- `track_b_baseline.py`: Baseline embedding generation with `all-MiniLM-L6-v2`
- `track_b_finetuning.py`: Triplet-loss fine-tuning experiment and embedding export

## Usage
These scripts were exported from notebooks and contain Colab/hardcoded paths.
Update paths and environment-specific commands before running.

```bash
python Approach-1_Track-B/track_b_baseline.py
python Approach-1_Track-B/track_b_finetuning.py
```

## Notes
- Use these for quick experimentation only.
- For maintained Track B-style embedding export in this repo, use root `generate_embeddings.py`.
