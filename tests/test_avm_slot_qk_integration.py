import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.avm import ground_aerial_slot_target
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

    def encode_text(self, text):
        return self.text_feats


class SlotQkIntegrationTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(23)
        self.text_feats = torch.randn(4, 5, 8)
        self.images = torch.randn(4, 5, 8)
        self.ground_images = torch.randn(4, 5, 8)
        self.caption_ids = torch.tensor([
            [1, 2, 9, 0, 0],
            [2, 3, 9, 0, 0],
            [3, 4, 9, 0, 0],
            [4, 5, 9, 0, 0],
        ])
        self.batch = {
            "images": self.images,
            "ground_imgs": self.ground_images,
            "caption_ids": self.caption_ids,
            "pids": torch.arange(4),
        }

    def _build_model(
        self,
        supervision="qk",
        temperature=0.5,
        mask_loss_weight=1.0,
        avm_mode="slot",
    ):
        args = SimpleNamespace(
            loss_names="cda",
            pretrain_choice="dummy",
            img_size=(1, 1),
            stride_size=1,
            temperature=0.02,
            avm_mode=avm_mode,
            avm_num_slots=8,
            avm_loss_weight=1.0,
            avm_supervision=supervision,
            avm_qk_temperature=temperature,
            avm_mask_loss_weight=mask_loss_weight,
        )
        dummy_base = DummyBaseModel(self.text_feats)

        with patch(
            "model.build_finetune."
            "build_CLIP_from_openai_pretrained",
            return_value=(dummy_base, {"embed_dim": 8}),
        ):
            return IRRA(args)

    def _forward(self, model):
        with patch(
            "model.build_finetune.torch.autocast",
            side_effect=lambda *args, **kwargs: nullcontext(),
        ):
            return model(self.batch)

    def test_qk_forward_matches_weighted_bce(self):
        model = self._build_model(
            temperature=0.5, mask_loss_weight=0.7
        )
        ret = self._forward(model)

        aerial_slots = model.slot_pool(
            self.images[:, 1:, :]
        )
        ground_slots = model.slot_pool(
            self.ground_images[:, 1:, :]
        )
        mask = model.avm_mask_head(
            self.images[:, 0, :]
        )
        target = ground_aerial_slot_target(
            ground_slots, aerial_slots, temperature=0.5
        )
        expected_bce = F.binary_cross_entropy(mask, target)

        self.assertEqual(
            set(ret),
            {
                "cda_loss",
                "avm_ret_loss",
                "avm_mask_loss",
                "avm_mask_bce",
                "avm_mask_mean",
                "avm_mask_std",
                "avm_q_mean",
                "avm_q_std",
                "avm_q_sat",
            },
        )
        self.assertTrue(
            torch.allclose(
                ret["avm_mask_bce"], expected_bce, atol=1e-7
            )
        )
        self.assertTrue(
            torch.allclose(
                ret["avm_mask_loss"],
                0.7 * expected_bce,
                atol=1e-7,
            )
        )
        self.assertTrue(
            torch.allclose(
                ret["avm_q_mean"], target.mean(), atol=1e-7
            )
        )
        self.assertTrue(
            torch.allclose(
                ret["avm_q_std"],
                target.std(unbiased=False),
                atol=1e-7,
            )
        )

    def test_qk_mask_loss_only_updates_mask_prediction_path(self):
        model = self._build_model(
            temperature=0.5, mask_loss_weight=1.0
        )
        ret = self._forward(model)

        model.zero_grad(set_to_none=True)
        ret["avm_mask_loss"].backward()

        mask_grad = model.avm_mask_head.mlp[-1].weight.grad
        self.assertIsNotNone(mask_grad)
        self.assertTrue(torch.isfinite(mask_grad).all())
        self.assertGreater(mask_grad.abs().sum().item(), 0.0)
        self.assertIsNone(model.slot_pool.slot_queries.grad)

    def test_qk_can_run_as_zero_weight_diagnostic(self):
        model = self._build_model(
            temperature=0.5, mask_loss_weight=0.0
        )
        ret = self._forward(model)

        self.assertEqual(ret["avm_mask_loss"].item(), 0.0)
        self.assertGreater(ret["avm_mask_bce"].item(), 0.0)
        self.assertGreaterEqual(ret["avm_q_mean"].item(), 0.0)
        self.assertLessEqual(ret["avm_q_mean"].item(), 1.0)

    def test_no_qk_preserves_existing_slot_outputs(self):
        model = self._build_model(
            supervision="none",
            temperature=None,
            mask_loss_weight=0.0,
        )
        ret = self._forward(model)

        self.assertEqual(
            set(ret),
            {
                "cda_loss",
                "avm_ret_loss",
                "avm_mask_mean",
                "avm_mask_std",
            },
        )

    def test_qk_requires_explicit_positive_temperature(self):
        for temperature in (None, 0.0, -0.1):
            with self.subTest(temperature=temperature):
                with self.assertRaisesRegex(
                    ValueError, "positive avm_qk_temperature"
                ):
                    self._build_model(temperature=temperature)

    def test_qk_requires_slot_mode(self):
        with self.assertRaisesRegex(
            ValueError, "requires avm_mode='slot'"
        ):
            self._build_model(avm_mode="none")

    def test_qk_does_not_change_inference_interfaces(self):
        model = self._build_model()

        encoded_text = model.encode_text(self.caption_ids)
        encoded_gallery = model.encode_image(self.images)

        self.assertEqual(encoded_text.shape, (4, 8, 8))
        self.assertEqual(encoded_gallery.shape, (4, 8, 8))


if __name__ == "__main__":
    unittest.main()
