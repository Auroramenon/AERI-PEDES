import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from model.build_finetune import IRRA


class DummyBaseModel(nn.Module):
    """Minimal CLIP replacement for CPU integration tests."""

    def __init__(self, text_feats):
        super().__init__()
        self.text_feats = text_feats
        self.proj = nn.Linear(8, 8)

    def forward(self, images, ground_images, caption_ids):
        return images, ground_images, self.text_feats

    def encode_image(self, image):
        return image

    def encode_text(self, text):
        return self.text_feats


class SlotDecorrelationIntegrationTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(22)
        self.text_feats = torch.randn(4, 5, 8)
        self.images = torch.randn(4, 5, 8)
        self.caption_ids = torch.tensor([
            [1, 2, 9, 0, 0],
            [2, 3, 9, 0, 0],
            [3, 4, 9, 0, 0],
            [4, 5, 9, 0, 0],
        ])
        self.batch = {
            "images": self.images,
            "ground_imgs": torch.randn(4, 5, 8),
            "caption_ids": self.caption_ids,
            "pids": torch.arange(4),
        }

    def build_model(self, div_weight=None):
        dummy_base = DummyBaseModel(self.text_feats)
        args = SimpleNamespace(
            loss_names="cda",
            pretrain_choice="dummy",
            img_size=(1, 1),
            stride_size=1,
            temperature=0.02,
            avm_mode="slot",
            avm_num_slots=8,
            avm_loss_weight=1.0,
        )
        if div_weight is not None:
            args.avm_div_loss_weight = div_weight

        with patch(
            "model.build_finetune."
            "build_CLIP_from_openai_pretrained",
            return_value=(dummy_base, {"embed_dim": 8}),
        ):
            return IRRA(args)

    def run_forward(self, model):
        with patch(
            "model.build_finetune.torch.autocast",
            side_effect=lambda *args, **kwargs: nullcontext(),
        ):
            return model(self.batch)

    def test_zero_weight_preserves_original_outputs(self):
        model = self.build_model(div_weight=0.0)
        ret = self.run_forward(model)

        self.assertEqual(
            set(ret),
            {
                "cda_loss",
                "avm_ret_loss",
                "avm_mask_mean",
                "avm_mask_std",
            },
        )

    def test_positive_weight_adds_diversity_loss(self):
        model = self.build_model(div_weight=0.1)
        ret = self.run_forward(model)

        self.assertIn("avm_div_loss", ret)
        self.assertIn("avm_div_raw", ret)
        self.assertTrue(torch.isfinite(ret["avm_div_loss"]))
        self.assertTrue(torch.isfinite(ret["avm_div_raw"]))
        self.assertTrue(torch.allclose(
            ret["avm_div_loss"],
            0.1 * ret["avm_div_raw"],
            atol=1e-6,
        ))

        ret["avm_div_loss"].backward(retain_graph=True)
        query_grad = model.slot_pool.slot_queries.grad
        mask_grad = model.avm_mask_head.mlp[-1].weight.grad
        self.assertIsNotNone(query_grad)
        self.assertTrue(torch.isfinite(query_grad).all())
        self.assertGreater(query_grad.abs().sum().item(), 0.0)
        self.assertIsNone(mask_grad)

        model.zero_grad(set_to_none=True)
        total_loss = sum(
            value for key, value in ret.items()
            if "loss" in key
        )
        total_loss.backward()

        mask_grad = model.avm_mask_head.mlp[-1].weight.grad
        self.assertIsNotNone(mask_grad)
        self.assertTrue(torch.isfinite(mask_grad).all())
        self.assertGreater(mask_grad.abs().sum().item(), 0.0)

    def test_negative_weight_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "avm_div_loss_weight"
        ):
            self.build_model(div_weight=-0.1)


if __name__ == "__main__":
    unittest.main()
