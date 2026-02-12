import argparse
import yaml
import json
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaV2Tokenizer
from tqdm import tqdm

from models.deberta_ranker import DebertaRanker


class EmbeddingDataset(Dataset):
    """Dataset for extracting embeddings from text"""
    def __init__(self, df, tokenizer, max_len: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def _tokenize(self, text):
        return self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["text"])
        
        tok = self._tokenize(text)
        
        return {
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
            "idx": idx
        }


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
        help="Path to testb.jsonl file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="submission_trackb.jsonl",
        help="Path to output submission file"
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
        help="Use all 5 fold models for ensemble embeddings"
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def extract_embeddings_single(model, loader, device):
    """Extract embeddings from a single model"""
    model.eval()
    all_embeddings = []
    
    for batch in tqdm(loader, desc="Extracting embeddings"):
        embeddings = model.encode(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device)
        )
        all_embeddings.append(embeddings.cpu())
    
    return torch.cat(all_embeddings, dim=0)


@torch.no_grad()
def extract_embeddings_ensemble(models, loader, device):
    """Extract ensemble embeddings from multiple models"""
    all_fold_embeddings = []
    
    for i, model in enumerate(models):
        model.eval()
        fold_embeddings = []
        
        for batch in tqdm(loader, desc=f"Fold {i+1} embeddings"):
            embeddings = model.encode(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device)
            )
            fold_embeddings.append(embeddings.cpu())
        
        all_fold_embeddings.append(torch.cat(fold_embeddings, dim=0))
    
    # Average embeddings across all folds
    ensemble_embeddings = torch.stack(all_fold_embeddings).mean(dim=0)
    
    return ensemble_embeddings


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
    tokenizer = DebertaV2Tokenizer.from_pretrained(cfg["model"]["name"])
    
    # Create dataset
    test_dataset = EmbeddingDataset(
        test_df,
        tokenizer,
        cfg["model"]["max_len"]
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
            model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
            model.to(device)
            model.eval()
            models.append(model)
        
        print("Extracting ensemble embeddings...")
        embeddings = extract_embeddings_ensemble(models, test_loader, device)
        
    else:
        # Load single best model (fold 0)
        checkpoint_path = f"{args.checkpoint_dir}/best_fold_0.pt"
        print(f"Loading model from {checkpoint_path}")
        
        model = DebertaRanker(
            cfg["model"]["name"],
            freeze_layers=cfg["model"]["freeze_layers"]
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        model.to(device)
        
        print("Extracting embeddings...")
        embeddings = extract_embeddings_single(model, test_loader, device)
    
    # Convert to numpy and ensure all embeddings have same dimension
    embeddings_np = embeddings.numpy()
    embedding_dim = embeddings_np.shape[1]
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Total embeddings: {len(embeddings_np)}")
    
    # Write embeddings to output file in Track-B format
    print(f"Writing embeddings to {args.output_file}")
    with open(args.output_file, 'w') as f:
        for i in range(len(embeddings_np)):
            output = {
                "embedding": embeddings_np[i].tolist()
            }
            f.write(json.dumps(output) + '\n')
    
    print(f"Done! Embeddings saved to {args.output_file}")
    print(f"Format: {len(embeddings_np)} samples × {embedding_dim} dimensions")


if __name__ == "__main__":
    main()
