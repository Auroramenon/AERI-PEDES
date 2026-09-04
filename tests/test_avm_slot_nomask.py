import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from model.avm import slot_gallery_embedding, slot_scores
from model.build_finetune import IRRA
from utils.metrics import compute_retrieval_similarity


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


class SlotNoMaskTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(17)

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

    def build_model(self, mask_policy=None):
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
        if mask_policy is not None:
            args.avm_mask_policy = mask_policy

        with patch(
            "model.build_finetune."
            "build_CLIP_from_openai_pretrained",
            return_value=(dummy_base, {"embed_dim": 8}),
        ):
            return IRRA(args)

    def test_ones_policy_has_no_learned_mask_head(self):
        model = self.build_model(mask_policy="ones")

        self.assertIsNone(model.avm_mask_head)
        self.assertFalse(any(
            name.startswith("avm_mask_head")
            for name, _ in model.named_parameters()
        ))

        mask = model._build_slot_mask(self.images[:, 0, :])
        self.assertEqual(mask.shape, (4, 8))
        self.assertTrue(torch.equal(mask, torch.ones_like(mask)))

    def test_default_policy_remains_learned(self):
        model = self.build_model()

        self.assertEqual(model.avm_mask_policy, "learned")
        self.assertIsNotNone(model.avm_mask_head)
        mask = model._build_slot_mask(self.images[:, 0, :])
        self.assertTrue(
            torch.allclose(mask, torch.full_like(mask, 0.5))
        )

    def test_ones_policy_forward_reaches_slot_pool(self):
        model = self.build_model(mask_policy="ones")

        with patch(
            "model.build_finetune.torch.autocast",
            side_effect=lambda *args, **kwargs: nullcontext(),
        ):
            ret = model(self.batch)

        self.assertEqual(
            set(ret),
            {
                "cda_loss",
                "avm_ret_loss",
                "avm_mask_mean",
                "avm_mask_std",
            },
        )
        self.assertAlmostEqual(
            ret["avm_mask_mean"].item(), 1.0, places=7
        )
        self.assertAlmostEqual(
            ret["avm_mask_std"].item(), 0.0, places=7
        )

        total_loss = ret["cda_loss"] + ret["avm_ret_loss"]
        total_loss.backward()

        query_grad = model.slot_pool.slot_queries.grad
        self.assertIsNotNone(query_grad)
        self.assertTrue(torch.isfinite(query_grad).all())
        self.assertGreater(query_grad.abs().sum().item(), 0.0)

    def test_encode_paths_match_unmasked_gallery_formula(self):
        model = self.build_model(mask_policy="ones")

        encoded_text = model.encode_text(self.caption_ids)
        encoded_gallery = model.encode_image(self.images)

        text_slots = model.slot_pool(
            self.text_feats,
            valid_mask=self.caption_ids.ne(0),
        )
        aerial_slots = model.slot_pool(self.images[:, 1:, :])
        ones = torch.ones(4, 8)
        expected_gallery = slot_gallery_embedding(
            aerial_slots, ones
        )
        expected_scores = slot_scores(
            text_slots, aerial_slots, ones
        )
        evaluator_scores = compute_retrieval_similarity(
            encoded_text, encoded_gallery
        )

        self.assertTrue(torch.allclose(
            encoded_gallery, expected_gallery, atol=1e-6
        ))
        self.assertTrue(torch.allclose(
            evaluator_scores, expected_scores, atol=1e-6
        ))

    def test_invalid_mask_policy_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "avm_mask_policy"
        ):
            self.build_model(mask_policy="invalid")


if __name__ == "__main__":
    unittest.main()
