import argparse
import yaml
import json
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from transformers import DebertaV2Tokenizer
from tqdm import tqdm

from models.deberta_ranker import DebertaRanker
from utils.dataset import SemEvalDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Path to test.jsonl file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="predictions.jsonl",
        help="Path to output predictions file"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory containing fold checkpoints"
    )
    parser.add_argument(
        "--use_ensemble",
        action="store_true",
        help="Use all 5 fold models for ensemble predictions"
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def predict_with_model(model, loader, device):
    """Get predictions from a single model"""
    model.eval()
    predictions = []
    
    for batch in tqdm(loader, desc="Predicting"):
        # Encode anchor and both texts
        anc_emb = model.encode(
            batch["anc_ids"].to(device),
            batch["anc_mask"].to(device)
        )
        a_emb = model.encode(
            batch["a_ids"].to(device),
            batch["a_mask"].to(device)
        )
        b_emb = model.encode(
            batch["b_ids"].to(device),
            batch["b_mask"].to(device)
        )
        
        # Compute similarity scores
        score_a = nn.functional.cosine_similarity(anc_emb, a_emb)
        score_b = nn.functional.cosine_similarity(anc_emb, b_emb)
        
        # Predict: text_a is closer if score_a > score_b
        preds = (score_a > score_b).cpu().numpy()
        predictions.extend(preds)
    
    return predictions


def ensemble_predict(models, loader, device):
    """Get ensemble predictions from multiple models"""
    all_scores = []
    
    for i, model in enumerate(models):
        model.eval()
        fold_scores = []
        
        for batch in tqdm(loader, desc=f"Fold {i+1}"):
            anc_emb = model.encode(
                batch["anc_ids"].to(device),
                batch["anc_mask"].to(device)
            )
            a_emb = model.encode(
                batch["a_ids"].to(device),
                batch["a_mask"].to(device)
            )
            b_emb = model.encode(
                batch["b_ids"].to(device),
                batch["b_mask"].to(device)
            )
            
            score_a = nn.functional.cosine_similarity(anc_emb, a_emb)
            score_b = nn.functional.cosine_similarity(anc_emb, b_emb)
            
            # Store the difference: positive if text_a is closer
            diff = (score_a - score_b).detach().cpu().numpy()
            fold_scores.extend(diff)
        
        all_scores.append(fold_scores)
    
    # Average predictions across all folds
    import numpy as np
    avg_scores = np.mean(all_scores, axis=0)
    predictions = (avg_scores > 0).astype(bool)
    
    return predictions


def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test data
    print(f"Loading test data from {args.test_file}")
    test_df = pd.read_json(args.test_file, lines=True)
    print(f"Test samples: {len(test_df)}")
    
    # Initialize tokenizer
    tokenizer = DebertaV2Tokenizer.from_pretrained(
        cfg["model"]["name"]
    )
    
    # Create dataset (training=False to avoid data augmentation)
    test_dataset = SemEvalDataset(
        test_df,
        tokenizer,
        cfg["model"]["max_len"],
        training=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=0
    )
    
    if args.use_ensemble:
        # Load all fold models
        print("Loading ensemble models...")
        models = []
        for fold in range(cfg["training"]["num_folds"]):
            checkpoint_path = f"{args.checkpoint_dir}/best_fold_{fold}.pt"
            print(f"Loading {checkpoint_path}")
            
            model = DebertaRanker(
                cfg["model"]["name"],
                freeze_layers=cfg["model"]["freeze_layers"]
            )
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.to(device)
            model.eval()
            models.append(model)
        
        print("Running ensemble predictions...")
        predictions = ensemble_predict(models, test_loader, device)
        
    else:
        # Load single best model (fold 0 by default, or you can choose based on validation)
        checkpoint_path = f"{args.checkpoint_dir}/best_fold_0.pt"
        print(f"Loading model from {checkpoint_path}")
        
        model = DebertaRanker(
            cfg["model"]["name"],
            freeze_layers=cfg["model"]["freeze_layers"]
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        
        print("Running predictions...")
        predictions = predict_with_model(model, test_loader, device)
    
    # Write predictions to output file
    print(f"Writing predictions to {args.output_file}")
    with open(args.output_file, 'w') as f:
        for pred in predictions:
            f.write(json.dumps({"text_a_is_closer": bool(pred)}) + '\n')
    
    print(f"Done! Predictions saved to {args.output_file}")
    print(f"text_a predicted as closer: {sum(predictions)}/{len(predictions)} times")


if __name__ == "__main__":
    main()
