# SemEval 2026 Task 4
## Narrative Story Similarity & Representation Learning

This repository contains our experiments and models for **SemEval 2026 Task 4: Narrative Story Similarity and Narrative Representation Learning**.

The goal of this task is to develop systems capable of understanding deep narrative structure in stories rather than relying on surface-level lexical similarity.

Our approach explores transformer-based ranking models, embedding-based similarity methods, and hybrid semantic representations for comparing narrative plots extracted from Wikipedia summaries of books and films.

---

## Task Description

SemEval-2026 Task 4 focuses on identifying narrative similarity between stories.

Given a story (the **anchor**) and two candidate stories, the model must determine which candidate is narratively closer to the anchor.

Unlike traditional semantic similarity tasks, this task requires systems to understand story structure and narrative progression.

### What Is Narrative Similarity?

Narrative similarity is determined by three fundamental components:

| Component | Description |
|---|---|
| **Abstract Theme** | The central ideas, motivations, and themes of the story |
| **Course of Action** | The sequence of key events and narrative developments |
| **Outcomes** | The consequences or resolution of the narrative |

Systems that rely only on word overlap or lexical similarity generally perform poorly. Successful approaches must capture higher-level narrative semantics and structural similarity.

---

## Task Tracks

### Track A — Narrative Similarity Ranking

The system receives a triplet of stories — an anchor, Candidate A, and Candidate B — and must determine which candidate is more narratively similar to the anchor.

**Input Format**
```json
{
  "anchor_text": "...",
  "text_a": "...",
  "text_b": "...",
  "text_a_is_closer": true
}
```

**Output Format**
```json
{"text_a_is_closer": true}
```
or
```json
{"text_a_is_closer": false}
```

### Track B — Narrative Representation Learning

Track B requires systems to generate dense vector **embeddings** for stories that capture their narrative structure. These embeddings are evaluated by measuring how well similar stories are located close to each other in the embedding space. Evaluation is performed using ranking metrics based on cosine similarity between embeddings.

### Special Category

The task also includes a special category for **symbolic approaches**, such as:
- Narrative schemas
- Story grammars
- Narrative graphs

These systems are evaluated separately and are strongly encouraged by the organizers.

---

## Dataset

The dataset consists of Wikipedia plot summaries of books and films.

- **Format**: JSON Lines (`.jsonl`) — each line contains one JSON object

| Split | Size |
|---|---|
| Training | ~2,076 triplets |
| Validation | ~200 triplets |

---

## Evaluation Metrics

### Track A

Binary accuracy:

$$\text{Accuracy} = \frac{\text{correct predictions}}{\text{total triplets}}$$

The system is correct when it predicts the same candidate as the gold label.

### Track B

Ranking-based evaluation computed using cosine similarity between embeddings. The metric measures how well the embedding space preserves narrative similarity ordering.

---

## Our Results

### Embedding Baseline (BGE-Large)

| Model | Training | Accuracy |
|---|---|---|
| BGE-Large | Zero-shot cosine similarity | 63.5% |
| BGE-Large | Triplet loss fine-tuning (4 epochs) | 64.5% |

---

## Project Structure

```
.
├── train.py
├── predict.py
├── generate_embeddings.py
├── MODEL_INDEX.md
├── configs/
├── bge_large/
├── DBERTA/
├── DBERTA_SIAMESE/
├── Approach-1_Track-A/
└── Approach-1_Track-B/
```

### Key Directories

| Directory | Description |
|---|---|
| `./` | Main DeBERTa training and inference pipeline |
| `DBERTA/` | Standalone DeBERTa implementation |
| `bge_large/` | BGE baseline, cross-encoder, and E5 experiments |
| `Approach-1_Track-A/` | Early SBERT and hybrid TF-IDF baselines |
| `Approach-1_Track-B/` | Prototype embedding generation scripts |
| `DBERTA_SIAMESE/` | Siamese DeBERTa prototype implementation |

---

## Quick Start

### Train DeBERTa Model (Track A)

```bash
python train.py --config configs/config.yaml
```

### Predict Track A Labels

```bash
python predict.py \
  --config configs/config.yaml \
  --test_file data/test.jsonl \
  --output_file predictions.jsonl
```

### Generate Track B Embeddings

```bash
python generate_embeddings.py \
  --config configs/config.yaml \
  --test_file data/testb.jsonl \
  --output_file submission_trackb.jsonl
```

### Run BGE Baseline

```bash
python bge_large/evaluation_files/bge_baseline_eval.py \
  --train_file bge_large/data/train.jsonl \
  --dev_file bge_large/data/dev.jsonl \
  --output_dir bge_large/outputs/bge_baseline
```

---

## Model Selection Guide

| Goal | Recommended Pipeline |
|---|---|
| Train a strong Track A model | Root DeBERTa pipeline |
| Generate Track B embeddings | `generate_embeddings.py` |
| Run embedding baseline quickly | BGE baseline scripts in `bge_large/evaluation_files/` |
| Prototype experiments | `Approach-1_*` directories |

For a full list of models with status and commands, see [MODEL_INDEX.md](MODEL_INDEX.md).

---

## Documentation

For consistent documentation across model folders, use:

- `docs/MODEL_README_TEMPLATE.md`

This template standardizes documentation for model architecture, training procedure, hyperparameters, and evaluation results.

---

## Reproducibility

All experiments were conducted using:
- Python 3.10+
- PyTorch
- HuggingFace Transformers
- SentenceTransformers

Random seeds and training configurations are stored in the `configs/` directory.
