import torch
import torch.nn.functional as F


def contrastive_loss(anchor, pos, neg, temperature: float):
    # Normalize embeddings for stable cosine similarity
    anchor = F.normalize(anchor, p=2, dim=1)
    pos = F.normalize(pos, p=2, dim=1)
    neg = F.normalize(neg, p=2, dim=1)
    
    # Compute cosine similarities
    pos_sim = (anchor * pos).sum(dim=1) / temperature
    neg_sim = (anchor * neg).sum(dim=1) / temperature
    
    # Use log-sum-exp trick for numerical stability
    logits = torch.stack([pos_sim, neg_sim], dim=1)
    loss = F.cross_entropy(logits, torch.zeros(logits.size(0), dtype=torch.long, device=logits.device))
    
    return loss


def margin_ranking_loss(pos_score, neg_score, margin: float):
    return F.margin_ranking_loss(
        pos_score,
        neg_score,
        torch.ones_like(pos_score),
        margin=margin
    )
