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


class DualSlotBranchIntegrationTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(32)
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
            avm_plain_branch_ratio=0.5,
        )

        with patch(
            "model.build_finetune."
            "build_CLIP_from_openai_pretrained",
            return_value=(dummy_base, {"embed_dim": 8}),
        ):
            self.model = IRRA(args)

    def run_forward(self):
        with patch(
            "model.build_finetune.torch.autocast",
            side_effect=lambda *args, **kwargs: nullcontext(),
        ):
            return self.model(self.batch)

    def test_dual_loss_reaches_pool_and_mask_head(self):
        ret = self.run_forward()
        ret["avm_ret_loss"].backward()

        query_grad = self.model.slot_pool.slot_queries.grad
        mask_grad = self.model.avm_mask_head.mlp[-1].weight.grad
        self.assertIsNotNone(query_grad)
        self.assertIsNotNone(mask_grad)
        self.assertTrue(torch.isfinite(query_grad).all())
        self.assertTrue(torch.isfinite(mask_grad).all())
        self.assertGreater(query_grad.abs().sum().item(), 0.0)
        self.assertGreater(mask_grad.abs().sum().item(), 0.0)

    def test_evaluation_keeps_masked_offline_gallery(self):
        with torch.no_grad():
            self.model.avm_mask_head.mlp[-1].bias.copy_(
                torch.linspace(-2.0, 2.0, 8)
            )

        encoded_text = self.model.encode_text(self.caption_ids)
        encoded_gallery = self.model.encode_image(self.images)

        text_slots = self.model.slot_pool(
            self.text_feats,
            valid_mask=self.caption_ids.ne(0),
        )
        aerial_cls = self.images[:, 0, :].float()
        aerial_slots = self.model.slot_pool(
            self.images[:, 1:, :]
        )
        learned_mask = self.model.avm_mask_head(aerial_cls)
        expected_gallery = slot_gallery_embedding(
            aerial_slots, learned_mask
        )
        expected_scores = slot_scores(
            text_slots, aerial_slots, learned_mask
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


if __name__ == "__main__":
    unittest.main()
