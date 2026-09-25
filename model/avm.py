import math

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


# ---------------------------------------------------------------------------
# DPM port (Tan et al., ACM MM 2022; DPM++ arXiv 2605.06637).
#
# DPM predicts a mask from the incomplete input and applies it to the
# holistic side (the class prototypes), re-normalising afterwards
# (DPM-main/utils/metrics.py cosine_single, loss/metric_learning.py
# MArcface). Here the aerial image is the incomplete side and the text is
# the holistic side, so the aerial mask is applied to every text candidate:
#
#     s(T_j, A_i) = < normalize(m_i * t_j), a_i >
#                 = sum_d m_id t_jd a_id / || m_i * t_j ||
#
# with t and a unit vectors. Both the numerator and the squared
# denominator are plain matrix products, so the gallery side (m * a and
# m * m) can still be pre-computed offline.
# ---------------------------------------------------------------------------


def text_side_masked_scores(text_feats, aerial_feats, mask, eps=1e-6):
    """Return DPM-style masked scores with shape [B_text, B_aerial]."""
    text_unit = F.normalize(text_feats.float(), p=2, dim=-1, eps=eps)
    aerial_unit = F.normalize(aerial_feats.float(), p=2, dim=-1, eps=eps)
    mask = mask.float()

    numerator = text_unit @ (mask * aerial_unit).t()
    denominator = (text_unit * text_unit) @ (mask * mask).t()
    return numerator / denominator.clamp_min(eps * eps).sqrt()


def plain_cosine_scores(text_feats, aerial_feats, eps=1e-6):
    """Unmasked text-to-aerial cosine, i.e. the CFAN retrieval score."""
    text_unit = F.normalize(text_feats.float(), p=2, dim=-1, eps=eps)
    aerial_unit = F.normalize(aerial_feats.float(), p=2, dim=-1, eps=eps)
    return text_unit @ aerial_unit.t()


def dpm_retrieval_scores(text_feats, gallery_feats, embed_dim):
    """Score text queries against gallery rows packed as [aerial | mask]."""
    if gallery_feats.shape[-1] != 2 * embed_dim:
        raise ValueError(
            "DPM gallery features must be [aerial, mask] with width "
            f"{2 * embed_dim}, got {gallery_feats.shape[-1]}"
        )
    aerial, mask = gallery_feats.split(embed_dim, dim=-1)

    plain = plain_cosine_scores(text_feats, aerial)
    masked = text_side_masked_scores(text_feats, aerial, mask)
    return {"plain": plain, "masked": masked, "sum": plain + masked}


def participation_ratio(mask):
    """Per-image (sum m)^2 / (D * sum m^2): 1 for a uniform mask, lower when
    the mask concentrates on fewer channels. Invariant to mask scale, so it
    stays meaningful under the re-normalised score. Differentiable."""
    mask = mask.float()
    return mask.sum(dim=1).pow(2) / (
        mask.shape[1] * mask.pow(2).sum(dim=1)
    ).clamp_min(1e-12)


def mask_statistics(mask):
    """Collapse diagnostics that are meaningful under re-normalisation.

    A uniform mask (all 1, all 0.5, ...) gives exactly the plain cosine, so
    the useful signals are the spread across channels inside one image and
    the participation ratio.
    """
    mask = mask.detach().float()
    instance_std = mask.std(dim=1, unbiased=False).mean()
    return instance_std, participation_ratio(mask).mean()


# ---------------------------------------------------------------------------
# Idea 1: controlled occlusion instead of q_k.
# ---------------------------------------------------------------------------


def occlude_band(images, ratio):
    """Blank one random contiguous band covering `ratio` of axis -2.

    For [B, C, H, W] images axis -2 is the height, so each image loses a
    horizontal strip (e.g. head, torso or legs). The fill value 0 is the
    dataset mean colour after normalisation. Returns a new tensor.
    """
    length = images.shape[-2]
    band = max(1, int(round(ratio * length)))
    if band >= length:
        raise ValueError(f"occlusion ratio {ratio} blanks the whole image")

    top = torch.randint(0, length - band + 1, (images.shape[0], 1), device=images.device)
    rows = torch.arange(length, device=images.device).unsqueeze(0)
    blank = (rows >= top) & (rows < top + band)                 # [B, length]
    shape = [images.shape[0]] + [1] * (images.ndim - 1)
    shape[-2] = length
    return images * (~blank).reshape(shape).to(images.dtype)


# ---------------------------------------------------------------------------
# Idea 2: literal DPM masked ID loss (DPM-main loss/metric_learning.py
# MArcface). Identity prototypes are shared by aerial and text features;
# the aerial mask cuts the prototypes exactly as it cuts the text at test.
# ---------------------------------------------------------------------------


class PrototypeHead(nn.Module):
    """Identity prototypes W [num_ids, D], xavier init as in DPM MArcface."""

    def __init__(self, num_ids, embed_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_ids, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def plain_logits(self, feats):
        # DPM's plain branch: unnormalised linear softmax ("S" in Table 4).
        return feats.float() @ self.weight.float().t()

    def masked_cosine(self, aerial_feats, mask):
        # cos(x_i, normalize(m_i * w_c)) for every class c -> [B, num_ids].
        return text_side_masked_scores(self.weight, aerial_feats, mask).t()


def arcface_logits(cosine, labels, scale, margin):
    """ArcFace logits s * cos(theta + m) on the target class (DPM MArcface)."""
    cosine = cosine.clamp(-1.0, 1.0)
    sine = (1.0 - cosine.pow(2)).clamp_min(0.0).sqrt()
    phi = cosine * math.cos(margin) - sine * math.sin(margin)
    # Same fallback as DPM when theta + m would pass pi.
    phi = torch.where(
        cosine > math.cos(math.pi - margin),
        phi,
        cosine - math.sin(math.pi - margin) * margin,
    )
    one_hot = F.one_hot(labels.long(), cosine.shape[1]).to(cosine.dtype)
    return scale * (one_hot * phi + (1.0 - one_hot) * cosine)


class StaticMask(nn.Module):
    """One learned channel mask shared by every aerial image.

    Control for the instance-adaptive claim: same capacity to reweight
    channels, but no dependence on the image.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, batch_size):
        return torch.sigmoid(self.logits.float()).unsqueeze(0).expand(
            batch_size, -1
        )


def conv3x3_block(in_planes, out_planes):
    """Same block as DPM++ model/clip/model.py conv3x3_block."""
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class HierarchicalMaskGenerator(nn.Module):
    """DPM++ Hierarchical Mask Generator for the CLIP ViT-B/16 aerial encoder.

    Ported from DPM++ model/clip/model.py: patch tokens of blocks 2, 4, 10
    and 12 are concatenated along channels, passed through three conv
    blocks with max pooling, and a zero-initialised linear layer maps the
    pooled vector to a sigmoid mask (so every channel starts at 0.5).

    The only change is the patch grid: DPM++ assumes a 2:1 grid, CFAN uses
    384x128 inputs with stride 16, i.e. a 24x8 grid, passed in as grid_hw.
    """

    LAYERS = (1, 3, 9, 11)

    def __init__(self, width, out_dim, grid_hw):
        super().__init__()
        self.grid_hw = tuple(grid_hw)
        levels = len(self.LAYERS)

        self.conv = nn.Sequential(
            conv3x3_block(width * levels, width * 2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_block(width * 2, width),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv3x3_block(width, width),
            nn.AdaptiveMaxPool2d((1, 1)),
        )
        self.fc = nn.Linear(width, out_dim)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, hidden):
        """hidden: {layer index: [N + 1, B, width] tokens (LND, CLS first)}."""
        height, width = self.grid_hw
        maps = []
        for layer in self.LAYERS:
            tokens = hidden[layer][1:].float()          # drop CLS -> [N, B, C]
            if tokens.shape[0] != height * width:
                raise ValueError(
                    f"layer {layer} has {tokens.shape[0]} patch tokens, "
                    f"expected {height}x{width}"
                )
            maps.append(
                tokens.permute(1, 2, 0).reshape(
                    tokens.shape[1], tokens.shape[2], height, width
                )
            )
        pooled = self.conv(torch.cat(maps, dim=1)).flatten(1)
        return torch.sigmoid(self.fc(pooled))
