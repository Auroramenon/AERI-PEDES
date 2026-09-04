import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from model import objectives
from model.avm import slot_scores
from model.build_finetune import IRRA


class DummyBaseModel(nn.Module):
    """Minimal CLIP replacement for CPU tests."""

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


class DualSlotBranchTest(unittest.TestCase):

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

    def build_model(self, ratio=None):
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
        if ratio is not None:
            args.avm_plain_branch_ratio = ratio

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

    def test_zero_ratio_preserves_original_outputs(self):
        model = self.build_model(ratio=0.0)
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

    def test_half_ratio_matches_plain_masked_average(self):
        model = self.build_model(ratio=0.5)
        ret = self.run_forward(model)

        text_slots = model.slot_pool(
            self.text_feats,
            valid_mask=self.caption_ids.ne(0),
        )
        aerial_slots = model.slot_pool(self.images[:, 1:, :])
        learned_mask = model.avm_mask_head(
            self.images[:, 0, :].float()
        )
        masked_scores = slot_scores(
            text_slots, aerial_slots, learned_mask
        )
        plain_scores = slot_scores(
            text_slots,
            aerial_slots,
            torch.ones_like(learned_mask),
        )
        masked_ret = objectives.compute_sdm_from_scores(
            masked_scores,
            self.batch["pids"],
            model.logit_scale,
        )
        plain_ret = objectives.compute_sdm_from_scores(
            plain_scores,
            self.batch["pids"],
            model.logit_scale,
        )

        self.assertTrue(torch.allclose(
            ret["avm_ret_loss"],
            0.5 * plain_ret + 0.5 * masked_ret,
            atol=1e-6,
        ))
        self.assertTrue(torch.allclose(
            ret["avm_plain_ret"], plain_ret.detach(), atol=1e-6
        ))
        self.assertTrue(torch.allclose(
            ret["avm_masked_ret"],
            masked_ret.detach(),
            atol=1e-6,
        ))
        self.assertFalse(ret["avm_plain_ret"].requires_grad)
        self.assertFalse(ret["avm_masked_ret"].requires_grad)

    def test_invalid_ratio_is_rejected(self):
        for ratio in (-0.1, 1.1):
            with self.subTest(ratio=ratio):
                with self.assertRaisesRegex(
                    ValueError, "avm_plain_branch_ratio"
                ):
                    self.build_model(ratio=ratio)


if __name__ == "__main__":
    unittest.main()
