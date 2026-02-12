import argparse
import yaml
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

import pandas as pd
from sklearn.model_selection import KFold

from transformers import (
    DebertaV2Tokenizer,
    get_cosine_schedule_with_warmup
)

from models.deberta_ranker import DebertaRanker
from models.losses import contrastive_loss, margin_ranking_loss
from utils.dataset import SemEvalDataset
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config.yaml"
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    cfg,
    device
):
    model.train()
    total_loss = 0.0
    use_amp = device.type == 'cuda'

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)

        # Only use autocast for CUDA
        if use_amp:
            with autocast():
                anc = model.encode(
                    batch["anc_ids"].to(device),
                    batch["anc_mask"].to(device)
                )
                pos = model.encode(
                    batch["a_ids"].to(device),
                    batch["a_mask"].to(device)
                )
                neg = model.encode(
                    batch["b_ids"].to(device),
                    batch["b_mask"].to(device)
                )

                pos_score = nn.functional.cosine_similarity(anc, pos)
                neg_score = nn.functional.cosine_similarity(anc, neg)

                rank_loss = margin_ranking_loss(
                    pos_score,
                    neg_score,
                    cfg["training"]["margin"]
                )

                cont_loss = contrastive_loss(
                    anc,
                    pos,
                    neg,
                    cfg["training"]["temperature"]
                )

                loss = rank_loss + 0.5 * cont_loss
        else:
            anc = model.encode(
                batch["anc_ids"].to(device),
                batch["anc_mask"].to(device)
            )
            pos = model.encode(
                batch["a_ids"].to(device),
                batch["a_mask"].to(device)
            )
            neg = model.encode(
                batch["b_ids"].to(device),
                batch["b_mask"].to(device)
            )

            pos_score = nn.functional.cosine_similarity(anc, pos)
            neg_score = nn.functional.cosine_similarity(anc, neg)

            rank_loss = margin_ranking_loss(
                pos_score,
                neg_score,
                cfg["training"]["margin"]
            )

            cont_loss = contrastive_loss(
                anc,
                pos,
                neg,
                cfg["training"]["temperature"]
            )

            loss = rank_loss + 0.5 * cont_loss

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        del anc, pos, neg
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0

    for batch in loader:
        s1 = nn.functional.cosine_similarity(
            model.encode(
                batch["anc_ids"].to(device),
                batch["anc_mask"].to(device)
            ),
            model.encode(
                batch["a_ids"].to(device),
                batch["a_mask"].to(device)
            ),
        )

        s2 = nn.functional.cosine_similarity(
            model.encode(
                batch["anc_ids"].to(device),
                batch["anc_mask"].to(device)
            ),
            model.encode(
                batch["b_ids"].to(device),
                batch["b_mask"].to(device)
            ),
        )

        preds = (s1 > s2).long()
        correct += (preds.cpu() == batch["label"]).sum().item()
        total += preds.size(0)

    return correct / total


def main():
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["seed"])

    device = torch.device(
        cfg["device"] if torch.cuda.is_available() else "cpu"
    )
    print("Using device:", device)

    os.makedirs(cfg["paths"]["save_dir"], exist_ok=True)

    # Load data
    train_df = pd.read_json(cfg["paths"]["train_data"], lines=True)

    tokenizer = DebertaV2Tokenizer.from_pretrained(
        cfg["model"]["name"]
    )

    kf = KFold(
        n_splits=cfg["training"]["num_folds"],
        shuffle=True,
        random_state=cfg["seed"]
    )

    model_paths = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(train_df)):
        print(f"\n===== FOLD {fold + 1} =====")

        train_ds = SemEvalDataset(
            train_df.iloc[tr_idx],
            tokenizer,
            cfg["model"]["max_len"],
            training=True
        )

        val_ds = SemEvalDataset(
            train_df.iloc[val_idx],
            tokenizer,
            cfg["model"]["max_len"],
            training=False
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=True,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["training"]["batch_size"],
            pin_memory=True
        )

        model = DebertaRanker(
            cfg["model"]["name"],
            cfg["model"]["freeze_layers"]
        ).to(device)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"]
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(len(train_loader) * cfg["training"]["epochs"] * cfg["training"]["warmup_ratio"]),
            num_training_steps=len(train_loader) * cfg["training"]["epochs"]
        )

        # Only create GradScaler for CUDA to avoid warnings
        scaler = GradScaler() if device.type == 'cuda' else None
        best_acc = 0.0
        patience_counter = 0
        best_epoch = 0

        for epoch in range(cfg["training"]["epochs"]):
            loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                cfg,
                device
            )

            acc = evaluate(model, val_loader, device)

            print(
                f"Epoch {epoch + 1} | "
                f"Loss {loss:.4f} | "
                f"Val Acc {acc:.4f}"
            )

            # Early stopping with patience
            if acc > best_acc + cfg["training"]["min_delta"]:
                best_acc = acc
                best_epoch = epoch + 1
                patience_counter = 0
                path = os.path.join(
                    cfg["paths"]["save_dir"],
                    f"best_fold_{fold}.pt"
                )
                torch.save(model.state_dict(), path)
                print(f"✓ New best model saved (Val Acc: {best_acc:.4f})")
            else:
                patience_counter += 1
                print(f"No improvement ({patience_counter}/{cfg['training']['patience']})")
                
                if patience_counter >= cfg["training"]["patience"]:
                    print(f"Early stopping at epoch {epoch + 1}. Best was epoch {best_epoch} with Val Acc {best_acc:.4f}")
                    break

        model_paths.append(
            os.path.join(
                cfg["paths"]["save_dir"],
                f"best_fold_{fold}.pt"
            )
        )

        del model
        torch.cuda.empty_cache()

    print("Training complete.")
    print("Saved models:", model_paths)


if __name__ == "__main__":
    main()
