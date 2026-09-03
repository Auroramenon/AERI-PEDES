import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from model.avm import slot_gallery_embedding, slot_scores
from model.build_finetune import IRRA
from solver.build import build_optimizer
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


class SlotIntegrationTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(11)

        self.text_feats = torch.randn(4, 5, 8)
        dummy_base = DummyBaseModel(self.text_feats)
        self.args = SimpleNamespace(
            loss_names="cda",
            pretrain_choice="dummy",
            img_size=(1, 1),
            stride_size=1,
            temperature=0.02,
            avm_mode="slot",
            avm_num_slots=8,
            avm_loss_weight=1.0,
        )

        with patch(
            "model.build_finetune."
            "build_CLIP_from_openai_pretrained",
            return_value=(dummy_base, {"embed_dim": 8}),
        ):
            self.model = IRRA(self.args)

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

    def test_forward_runs_cda_and_slot_mask(self):
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

        query_grad = self.model.slot_pool.slot_queries.grad
        mask_grad = self.model.avm_mask_head.mlp[-1].weight.grad

        self.assertIsNotNone(query_grad)
        self.assertIsNotNone(mask_grad)
        self.assertGreater(query_grad.abs().sum().item(), 0.0)
        self.assertGreater(mask_grad.abs().sum().item(), 0.0)

    def test_encode_paths_match_offline_gallery_formula(self):
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
        mask = self.model.avm_mask_head(aerial_cls)
        expected_gallery = slot_gallery_embedding(
            aerial_slots, mask
        )
        expected_scores = slot_scores(
            text_slots, aerial_slots, mask
        )
        evaluator_scores = compute_retrieval_similarity(
            encoded_text, encoded_gallery
        )

        self.assertEqual(encoded_text.shape, (4, 8, 8))
        self.assertEqual(encoded_gallery.shape, (4, 8, 8))
        self.assertTrue(
            torch.allclose(
                encoded_gallery, expected_gallery, atol=1e-6
            )
        )
        self.assertTrue(
            torch.allclose(
                evaluator_scores, expected_scores, atol=1e-6
            )
        )

    def test_optimizer_assigns_avm_learning_rate(self):
        optimizer_args = SimpleNamespace(
            lr=5e-6,
            lr2=5e-5,
            lr_factor=5.0,
            avm_lr=1e-4,
            weight_decay=4e-5,
            optimizer="Adam",
            alpha=0.9,
            beta=0.999,
        )
        optimizer = build_optimizer(
            optimizer_args, self.model
        )
        parameter_lrs = {
            id(group["params"][0]): group["lr"]
            for group in optimizer.param_groups
        }
        trainable_parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]

        self.assertEqual(
            len(parameter_lrs), len(trainable_parameters)
        )

        for name, parameter in self.model.named_parameters():
            expected_lr = (
                optimizer_args.avm_lr
                if name.startswith(("slot_pool", "avm_mask_head"))
                else optimizer_args.lr
            )
            self.assertEqual(parameter_lrs[id(parameter)], expected_lr)


if __name__ == "__main__":
    unittest.main()
