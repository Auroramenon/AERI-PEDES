import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureMaskHead(nn.Module):
    """Predict an aerial-instance-conditioned feature mask."""

    def __init__(self, embed_dim, hidden_dim=None):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = embed_dim * 2

        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Initial mask = sigmoid(0) = 0.5.
        # With post-mask normalization, the initial retrieval score
        # is equivalent to the original unmasked cosine score.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, aerial_cls):
        return torch.sigmoid(self.mlp(aerial_cls))


def feature_gallery_embedding(aerial_cls, mask, eps=1e-6):
    """Build the query-independent masked aerial gallery embedding."""
    aerial_unit = F.normalize(aerial_cls.float(), p=2, dim=-1, eps=eps)
    masked_aerial = aerial_unit * mask.float()

    # Remove the meaningless common mask-amplitude/temperature effect,
    # while preserving relative channel reweighting.
    return F.normalize(masked_aerial, p=2, dim=-1, eps=eps)


def feature_scores(text_cls, aerial_cls, mask, eps=1e-6):
    """Return raw text-to-aerial cosine scores with shape [B_text, B_aerial]."""
    text_unit = F.normalize(text_cls.float(), p=2, dim=-1, eps=eps)
    gallery_unit = feature_gallery_embedding(aerial_cls, mask, eps=eps)
    return text_unit @ gallery_unit.t()
