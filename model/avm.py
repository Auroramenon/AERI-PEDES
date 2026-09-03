import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticSlotPool(nn.Module):
    """Pool a token sequence with shared learnable semantic queries."""

    def __init__(self, embed_dim, num_slots=8):
        super().__init__()

        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if num_slots <= 0:
            raise ValueError("num_slots must be positive")

        self.embed_dim = embed_dim
        self.num_slots = num_slots
        self.slot_queries = nn.Parameter(
            torch.empty(num_slots, embed_dim)
        )

        # Standard scaled dot-product attention assumes unit-variance
        # query/key components before the 1/sqrt(d) scale.
        nn.init.normal_(self.slot_queries, mean=0.0, std=1.0)
        self.scale = 1.0 / math.sqrt(embed_dim)

    def forward(self, tokens, valid_mask=None):
        """
        Args:
            tokens: token or patch features with shape [B, L, D].
            valid_mask: optional valid-token mask with shape [B, L].

        Returns:
            Semantic slots with shape [B, K, D].
        """
        if tokens.ndim != 3:
            raise ValueError(
                f"tokens must be [B, L, D], got {tokens.shape}"
            )
        if tokens.shape[-1] != self.embed_dim:
            raise ValueError(
                "token dimension does not match slot query dimension"
            )

        tokens = tokens.float()
        batch_size = tokens.shape[0]
        queries = self.slot_queries.float().unsqueeze(0).expand(
            batch_size, -1, -1
        )

        attention_logits = torch.einsum(
            "bkd,bld->bkl", queries, tokens
        ) * self.scale

        if valid_mask is not None:
            if valid_mask.shape != tokens.shape[:2]:
                raise ValueError(
                    "valid_mask must match the token dimensions [B, L]"
                )
            valid_mask = valid_mask.to(
                device=tokens.device, dtype=torch.bool
            )
            if not torch.all(valid_mask.any(dim=1)):
                raise ValueError(
                    "each sample must contain at least one valid token"
                )
            attention_logits = attention_logits.masked_fill(
                ~valid_mask.unsqueeze(1),
                torch.finfo(attention_logits.dtype).min,
            )

        attention = F.softmax(attention_logits, dim=-1)
        return torch.einsum("bkl,bld->bkd", attention, tokens)


class SlotMaskHead(nn.Module):
    """Predict an aerial-instance-conditioned mask over semantic slots."""

    def __init__(self, embed_dim, num_slots=8, hidden_dim=None):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = embed_dim * 2

        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_slots),
        )

        # Initial slot mask = sigmoid(0) = 0.5.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, aerial_cls):
        return torch.sigmoid(self.mlp(aerial_cls.float()))


def slot_gallery_embedding(aerial_slots, mask, eps=1e-6):
    """Create the query-independent masked aerial gallery slots."""
    if aerial_slots.ndim != 3:
        raise ValueError(
            "aerial_slots must have shape [B, K, D]"
        )
    if mask.shape != aerial_slots.shape[:2]:
        raise ValueError("mask must have shape [B, K]")

    aerial_unit = F.normalize(
        aerial_slots.float(), p=2, dim=-1, eps=eps
    )

    # Do not normalize again after multiplying by the scalar slot mask.
    # Otherwise every non-zero scalar mask would be cancelled.
    return mask.float().unsqueeze(-1) * aerial_unit


def batched_slot_dot(text_slots, masked_gallery_slots, eps=1e-6):
    """Score text slots against precomputed masked gallery slots."""
    if text_slots.ndim != 3 or masked_gallery_slots.ndim != 3:
        raise ValueError(
            "slot tensors must have shape [B, K, D]"
        )
    if text_slots.shape[1:] != masked_gallery_slots.shape[1:]:
        raise ValueError(
            "text and gallery slots must share [K, D] dimensions"
        )

    text_unit = F.normalize(
        text_slots.float(), p=2, dim=-1, eps=eps
    )
    per_slot_scores = torch.einsum(
        "qkd,gkd->qgk",
        text_unit,
        masked_gallery_slots.float(),
    )
    return per_slot_scores.sum(dim=-1)


def slot_scores(text_slots, aerial_slots, mask, eps=1e-6):
    """Return raw s(T,A) = sum_k m_k cos(z_k^T, z_k^A)."""
    masked_gallery = slot_gallery_embedding(
        aerial_slots, mask, eps=eps
    )
    return batched_slot_dot(
        text_slots, masked_gallery, eps=eps
    )
