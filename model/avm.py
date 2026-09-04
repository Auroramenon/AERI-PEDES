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


class CLS2MaskFeatureCrossAttention(nn.Module):
    """Refine an image CLS token with image-side mask features."""

    def __init__(
        self,
        embed_dim,
        num_heads=None,
        residual_scale=0.1,
    ):
        super().__init__()

        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if num_heads is None:
            num_heads = max(1, embed_dim // 64)
        if num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError(
                "num_heads must be positive and divide embed_dim"
            )
        if residual_scale < 0:
            raise ValueError("residual_scale must be non-negative")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # Keep the first SMCA ablation fixed and close to the pretrained CLS.
        self.residual_scale = float(residual_scale)

        self.query_norm = nn.LayerNorm(embed_dim)
        self.mask_feature_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(
        self,
        image_cls,
        mask_features,
        return_attention=False,
    ):
        """
        Args:
            image_cls: global image features with shape [B, D].
            mask_features: image-side mask features with shape [B, K, D].
            return_attention: whether to return per-head CLS-to-feature weights.

        Returns:
            Enhanced image CLS with shape [B, D]. If requested, also returns
            attention weights with shape [B, H, 1, K].
        """
        if image_cls.ndim != 2:
            raise ValueError(
                f"image_cls must be [B, D], got {image_cls.shape}"
            )
        if mask_features.ndim != 3:
            raise ValueError(
                "mask_features must have shape [B, K, D]"
            )
        if image_cls.shape[0] != mask_features.shape[0]:
            raise ValueError(
                "image_cls and mask_features batch dimensions must match"
            )
        if (
            image_cls.shape[-1] != self.embed_dim
            or mask_features.shape[-1] != self.embed_dim
        ):
            raise ValueError(
                "image_cls and mask_features must match embed_dim"
            )

        image_cls = image_cls.float()
        query = self.query_norm(image_cls).unsqueeze(1)
        key_value = self.mask_feature_norm(mask_features.float())

        attention_output, attention_weights = self.cross_attn(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        enhanced_cls = (
            image_cls
            + self.residual_scale * attention_output[:, 0, :]
        )

        if return_attention:
            return enhanced_cls, attention_weights
        return enhanced_cls


def smca_diagnostics(
    image_cls,
    enhanced_cls,
    mask_features,
    attention_weights,
    eps=1e-8,
):
    """Return detached collapse and branch-usage diagnostics for SMCA."""
    if mask_features.ndim != 3:
        raise ValueError("mask_features must have shape [B, K, D]")
    if attention_weights.ndim != 4:
        raise ValueError(
            "attention_weights must have shape [B, H, 1, K]"
        )

    num_features = mask_features.shape[1]
    feature_unit = F.normalize(
        mask_features.float(), p=2, dim=-1, eps=eps
    )
    feature_similarity = torch.einsum(
        "bkd,bjd->bkj", feature_unit, feature_unit
    )
    if num_features > 1:
        off_diagonal = ~torch.eye(
            num_features,
            device=mask_features.device,
            dtype=torch.bool,
        )
        feature_abs_cosine = feature_similarity[
            :, off_diagonal
        ].abs().mean()
    else:
        feature_abs_cosine = feature_similarity.new_zeros(())

    attention = attention_weights.float().mean(dim=1).squeeze(1)
    attention = attention / attention.sum(
        dim=-1, keepdim=True
    ).clamp_min(eps)
    attention_entropy = -(
        attention * torch.log(attention.clamp_min(eps))
    ).sum(dim=-1)
    if num_features > 1:
        attention_entropy = attention_entropy / math.log(num_features)
    attention_entropy = attention_entropy.mean()

    delta_ratio = (
        (enhanced_cls.float() - image_cls.float()).norm(dim=-1)
        / image_cls.float().norm(dim=-1).clamp_min(eps)
    ).mean()

    return {
        "smca_feature_abs_cosine": feature_abs_cosine.detach(),
        "smca_attention_entropy": attention_entropy.detach(),
        "smca_delta_ratio": delta_ratio.detach(),
    }


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
