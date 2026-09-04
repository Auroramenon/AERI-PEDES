import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from model import objectives
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


class SlotDecorrelationLossTest(unittest.TestCase):

    def test_identical_slots_have_unit_penalty(self):
        base = torch.randn(2, 1, 8)
        slots = base.expand(-1, 4, -1).clone()

        loss = objectives.compute_slot_decorrelation_loss(slots)

        self.assertAlmostEqual(loss.item(), 1.0, places=6)

    def test_orthogonal_slots_have_zero_penalty(self):
        slots = torch.eye(4).unsqueeze(0)

        loss = objectives.compute_slot_decorrelation_loss(slots)

        self.assertAlmostEqual(loss.item(), 0.0, places=7)

    def test_loss_reaches_slot_tensor(self):
        slots = torch.randn(3, 4, 8, requires_grad=True)

        loss = objectives.compute_slot_decorrelation_loss(slots)
        loss.backward()

        self.assertIsNotNone(slots.grad)
        self.assertTrue(torch.isfinite(slots.grad).all())
        self.assertGreater(slots.grad.abs().sum().item(), 0.0)

    def test_single_slot_returns_differentiable_zero(self):
        slots = torch.randn(2, 1, 8, requires_grad=True)

        loss = objectives.compute_slot_decorrelation_loss(slots)
        loss.backward()

        self.assertEqual(loss.item(), 0.0)
        self.assertIsNotNone(slots.grad)
        self.assertTrue(torch.equal(
            slots.grad, torch.zeros_like(slots.grad)
        ))


class SlotAttentionDecorrelationLossTest(unittest.TestCase):

    def test_identical_attention_maps_have_unit_penalty(self):
        attention = torch.full((2, 4, 5), 0.2)

        loss = objectives.compute_attention_map_decorrelation_loss(
            attention
        )

        self.assertAlmostEqual(loss.item(), 1.0, places=6)

    def test_disjoint_attention_maps_have_zero_penalty(self):
        attention = torch.eye(4).unsqueeze(0)

        loss = objectives.compute_attention_map_decorrelation_loss(
            attention
        )

        self.assertAlmostEqual(loss.item(), 0.0, places=7)

    def test_loss_reaches_slot_queries_through_live_attention(self):
        from model.avm import SemanticSlotPool

        pool = SemanticSlotPool(embed_dim=8, num_slots=4)
        tokens = torch.randn(3, 5, 8)
        _, attention = pool(
            tokens,
            return_attention=True,
            detach_attention=False,
        )

        loss = objectives.compute_attention_map_decorrelation_loss(
            attention
        )
        loss.backward()

        self.assertTrue(attention.requires_grad)
        self.assertIsNotNone(pool.slot_queries.grad)
        self.assertTrue(torch.isfinite(pool.slot_queries.grad).all())
        self.assertGreater(
            pool.slot_queries.grad.abs().sum().item(), 0.0
        )


class SMCADecorrelationIntegrationTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(31)
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

    def build_model(self, div_weight, attn_div_weight=0.0):
        dummy_base = DummyBaseModel(self.text_feats)
        args = SimpleNamespace(
            loss_names="cda",
            pretrain_choice="dummy",
            img_size=(1, 1),
            stride_size=1,
            temperature=0.02,
            avm_mode="slot_cross",
            avm_num_slots=4,
            avm_mask_policy="none",
            avm_loss_weight=1.0,
            avm_div_loss_weight=div_weight,
            avm_attn_div_loss_weight=attn_div_weight,
        )

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

    def test_zero_weight_preserves_smca_outputs(self):
        ret = self.run_forward(self.build_model(div_weight=0.0))

        self.assertNotIn("smca_div_loss", ret)
        self.assertNotIn("smca_div_raw", ret)
        self.assertNotIn("smca_attn_div_loss", ret)
        self.assertNotIn("smca_attn_div_raw", ret)

    def test_positive_weight_adds_one_weighted_loss(self):
        model = self.build_model(div_weight=0.1)
        ret = self.run_forward(model)

        self.assertIn("smca_div_loss", ret)
        self.assertIn("smca_div_raw", ret)
        self.assertTrue(torch.allclose(
            ret["smca_div_loss"],
            0.1 * ret["smca_div_raw"],
            atol=1e-6,
        ))

        ret["smca_div_loss"].backward()
        slot_grad = model.slot_pool.slot_queries.grad
        cross_grad = model.smca_cross_attn.cross_attn.in_proj_weight.grad

        self.assertIsNotNone(slot_grad)
        self.assertTrue(torch.isfinite(slot_grad).all())
        self.assertGreater(slot_grad.abs().sum().item(), 0.0)
        self.assertIsNone(cross_grad)

    def test_attention_weight_adds_weighted_loss(self):
        model = self.build_model(
            div_weight=0.1,
            attn_div_weight=0.1,
        )
        ret = self.run_forward(model)

        self.assertIn("smca_attn_div_loss", ret)
        self.assertIn("smca_attn_div_raw", ret)
        self.assertTrue(torch.allclose(
            ret["smca_attn_div_loss"],
            0.1 * ret["smca_attn_div_raw"],
            atol=1e-6,
        ))

        ret["smca_attn_div_loss"].backward()
        slot_grad = model.slot_pool.slot_queries.grad
        cross_grad = model.smca_cross_attn.cross_attn.in_proj_weight.grad

        self.assertIsNotNone(slot_grad)
        self.assertTrue(torch.isfinite(slot_grad).all())
        self.assertGreater(slot_grad.abs().sum().item(), 0.0)
        self.assertIsNone(cross_grad)

    def test_negative_weight_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "avm_div_loss_weight"
        ):
            self.build_model(div_weight=-0.1)

    def test_negative_attention_weight_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "avm_attn_div_loss_weight"
        ):
            self.build_model(
                div_weight=0.1,
                attn_div_weight=-0.1,
            )


if __name__ == "__main__":
    unittest.main()
