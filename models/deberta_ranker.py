import torch
import torch.nn as nn
from transformers import DebertaV2Model


class MeanPooling(nn.Module):
    def forward(self, token_embeddings, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts


class DebertaRanker(nn.Module):
    def __init__(self, model_name: str, freeze_layers: int = 4):
        super().__init__()
        # Set use_safetensors to False to avoid loading error if .safetensors files are missing
        self.encoder = DebertaV2Model.from_pretrained(
            model_name, 
            use_safetensors=False,
            torch_dtype=torch.float32,  # Force float32 for CPU compatibility
            hidden_dropout_prob=0.2,  # Increased dropout
            attention_probs_dropout_prob=0.2
        )
        self.pool = MeanPooling()

        # Freeze early layers more aggressively
        for name, param in self.encoder.named_parameters():
            if any(f"layer.{i}." in name for i in range(freeze_layers)):
                param.requires_grad = False

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = self.pool(outputs.last_hidden_state, attention_mask)
        # Normalize embeddings for stable similarity computations
        return nn.functional.normalize(pooled, p=2, dim=1)
