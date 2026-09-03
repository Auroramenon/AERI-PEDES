import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.avm import feature_gallery_embedding
from model.build_finetune import IRRA


class DummyBaseModel(nn.Module):
    """Minimal CLIP replacement for CPU integration tests."""

    def __init__(self, text_feats):
        super().__init__()
        self.text_feats = text_feats

    def forward(self, images, ground_images, caption_ids):
        return images, ground_images, self.text_feats

    def encode_image(self, image):
        return image


class FeatureMaskIntegrationTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(11)

        self.text_feats = torch.randn(4, 3, 8)
        dummy_base = DummyBaseModel(self.text_feats)

        args = SimpleNamespace(
            loss_names="cda",
            pretrain_choice="dummy",
            img_size=(1, 1),
            stride_size=1,
            temperature=0.02,
            avm_mode="feature",
            avm_loss_weight=1.0,
        )

        with patch(
            "model.build_finetune."
            "build_CLIP_from_openai_pretrained",
            return_value=(dummy_base, {"embed_dim": 8}),
        ):
            self.model = IRRA(args)

        self.images = torch.randn(4, 3, 8)

        self.batch = {
            "images": self.images,
            "ground_imgs": torch.randn(4, 3, 8),
            "caption_ids": torch.tensor([
                [1, 2, 9],
                [2, 3, 9],
                [3, 4, 9],
                [4, 5, 9],
            ]),
            "pids": torch.arange(4),
        }

    def test_forward_runs_cda_and_feature_mask(self):
        with patch(
            "model.build_finetune.torch.autocast",
            side_effect=lambda *args, **kwargs: nullcontext(),
        ):
            ret = self.model(self.batch)

        self.assertEqual(
            set(ret),
            {
                "cda_loss",
                "avm_ret_loss",
                "avm_mask_mean",
                "avm_mask_std",
            },
        )

        self.assertTrue(torch.isfinite(ret["cda_loss"]))
        self.assertTrue(torch.isfinite(ret["avm_ret_loss"]))
        self.assertAlmostEqual(
            ret["avm_mask_mean"].item(), 0.5, places=7
        )
        self.assertAlmostEqual(
            ret["avm_mask_std"].item(), 0.0, places=7
        )

        total_loss = ret["cda_loss"] + ret["avm_ret_loss"]
        total_loss.backward()

        grad = self.model.avm_mask_head.mlp[-1].weight.grad

        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0.0)

    def test_encode_image_uses_masked_gallery_embedding(self):
        with torch.no_grad():
            self.model.avm_mask_head.mlp[-1].bias.copy_(
                torch.linspace(-2.0, 2.0, 8)
            )

        aerial_cls = self.images[:, 0, :].float()
        mask = self.model.avm_mask_head(aerial_cls)

        expected = feature_gallery_embedding(aerial_cls, mask)
        encoded = self.model.encode_image(self.images)
        baseline = F.normalize(aerial_cls, p=2, dim=-1)

        self.assertEqual(encoded.shape, (4, 8))
        self.assertTrue(
            torch.allclose(encoded, expected, atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(
                encoded.norm(dim=-1),
                torch.ones(4),
                atol=1e-6,
            )
        )

        # Non-uniform masks must alter the gallery direction.
        self.assertFalse(
            torch.allclose(encoded, baseline, atol=1e-6)
        )


if __name__ == "__main__":
    unittest.main()
