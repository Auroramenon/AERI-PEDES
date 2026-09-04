import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from model.avm import CLS2MaskFeatureCrossAttention
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


class CLS2MaskFeatureCrossAttentionTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(23)

    def test_cls_queries_multiple_mask_features(self):
        module = CLS2MaskFeatureCrossAttention(
            embed_dim=8,
            num_heads=2,
            residual_scale=0.1,
        )
        image_cls = torch.randn(3, 8)
        mask_features = torch.randn(3, 4, 8)

        enhanced, attention = module(
            image_cls,
            mask_features,
            return_attention=True,
        )

        self.assertEqual(enhanced.shape, (3, 8))
        self.assertEqual(attention.shape, (3, 2, 1, 4))
        self.assertTrue(torch.allclose(
            attention.sum(dim=-1),
            torch.ones(3, 2, 1),
            atol=1e-6,
        ))

    def test_fixed_small_residual_preserves_gradient_flow(self):
        module = CLS2MaskFeatureCrossAttention(
            embed_dim=8,
            num_heads=2,
            residual_scale=0.1,
        )
        image_cls = torch.randn(3, 8, requires_grad=True)
        mask_features = torch.randn(3, 4, 8, requires_grad=True)

        enhanced = module(image_cls, mask_features)
        enhanced.square().mean().backward()

        self.assertEqual(module.residual_scale, 0.1)
        self.assertIsNotNone(image_cls.grad)
        self.assertIsNotNone(mask_features.grad)
        self.assertGreater(mask_features.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(module.cross_attn.in_proj_weight.grad)
        self.assertGreater(
            module.cross_attn.in_proj_weight.grad.abs().sum().item(),
            0.0,
        )

    def test_rejects_reversed_or_mismatched_shapes(self):
        module = CLS2MaskFeatureCrossAttention(
            embed_dim=8,
            num_heads=2,
        )

        with self.assertRaisesRegex(ValueError, "image_cls"):
            module(torch.randn(3, 1, 8), torch.randn(3, 4, 8))

        with self.assertRaisesRegex(ValueError, "batch"):
            module(torch.randn(3, 8), torch.randn(2, 4, 8))


class SMCAIntegrationTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(29)

        self.text_feats = torch.randn(4, 5, 8)
        dummy_base = DummyBaseModel(self.text_feats)
        self.args = SimpleNamespace(
            loss_names="cda",
            pretrain_choice="dummy",
            img_size=(1, 1),
            stride_size=1,
            temperature=0.02,
            avm_mode="slot_cross",
            avm_num_slots=4,
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

    def test_forward_runs_cda_and_enhanced_sdm(self):
        with patch(
            "model.build_finetune.torch.autocast",
            side_effect=lambda *args, **kwargs: nullcontext(),
        ):
            ret = self.model(self.batch)

        self.assertEqual(
            set(ret),
            {
                "cda_loss",
                "smca_sdm_loss",
                "smca_feature_abs_cosine",
                "smca_attention_entropy",
                "smca_delta_ratio",
            },
        )
        for value in ret.values():
            self.assertTrue(torch.isfinite(value))

        total_loss = ret["cda_loss"] + ret["smca_sdm_loss"]
        total_loss.backward()

        slot_grad = self.model.slot_pool.slot_queries.grad
        cross_grad = self.model.smca_cross_attn.cross_attn.in_proj_weight.grad
        self.assertIsNotNone(slot_grad)
        self.assertIsNotNone(cross_grad)
        self.assertGreater(slot_grad.abs().sum().item(), 0.0)
        self.assertGreater(cross_grad.abs().sum().item(), 0.0)

    def test_encode_paths_use_global_text_and_enhanced_image(self):
        encoded_text = self.model.encode_text(self.caption_ids)
        encoded_image = self.model.encode_image(self.images)

        expected_text = self.text_feats[
            torch.arange(self.text_feats.shape[0]),
            self.caption_ids.argmax(dim=-1),
        ].float()
        image_cls = self.images[:, 0, :].float()
        mask_features = self.model.slot_pool(self.images[:, 1:, :])
        expected_image = self.model.smca_cross_attn(
            image_cls,
            mask_features,
        )

        self.assertEqual(encoded_text.shape, (4, 8))
        self.assertEqual(encoded_image.shape, (4, 8))
        self.assertTrue(torch.allclose(encoded_text, expected_text))
        self.assertTrue(torch.allclose(encoded_image, expected_image))

        similarity = compute_retrieval_similarity(
            encoded_text,
            encoded_image,
        )
        self.assertEqual(similarity.shape, (4, 4))

    def test_smca_parameters_use_avm_learning_rate(self):
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
        optimizer = build_optimizer(optimizer_args, self.model)
        parameter_lrs = {
            id(group["params"][0]): group["lr"]
            for group in optimizer.param_groups
        }

        for name, parameter in self.model.named_parameters():
            expected_lr = (
                optimizer_args.avm_lr
                if name.startswith(("slot_pool", "smca_cross_attn"))
                else optimizer_args.lr
            )
            self.assertEqual(parameter_lrs[id(parameter)], expected_lr)


if __name__ == "__main__":
    unittest.main()
