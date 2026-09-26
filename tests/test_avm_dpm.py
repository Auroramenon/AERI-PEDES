import logging
import re
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.avm import (
    FeatureMaskHead,
    HierarchicalMaskGenerator,
    PrototypeHead,
    StaticMask,
    arcface_logits,
    check_mask_groups,
    dpm_retrieval_scores,
    expand_groups,
    fixed_derangement,
    mask_statistics,
    occlude_band,
    participation_ratio,
    plain_cosine_scores,
    soft_row_correlation,
    text_side_masked_scores,
)
from model.build_finetune import IRRA
from model.clip_model import VisionTransformer
from model.objectives import compute_sdm_from_scores
from processor.processor_finetune import forward_losses, mask_generator_step
from solver import build_mask_optimizer, build_optimizer
from utils.metrics import Evaluator


GRID = (24, 8)
NUM_PATCHES = GRID[0] * GRID[1]
HIDDEN_WIDTH = 4
EMBED = 8


def no_autocast():
    return patch(
        "model.build_finetune.torch.autocast",
        side_effect=lambda *args, **kwargs: nullcontext(),
    )


class DPMScoreTest(unittest.TestCase):

    def test_masked_scores_match_per_pair_definition(self):
        torch.manual_seed(0)
        text, aerial = torch.randn(5, 16), torch.randn(3, 16)
        mask = torch.sigmoid(torch.randn(3, 16))

        scores = text_side_masked_scores(text, aerial, mask)

        expected = torch.empty(5, 3)
        for j in range(5):
            for i in range(3):
                masked_text = F.normalize(mask[i] * F.normalize(text[j], dim=0), dim=0)
                expected[j, i] = masked_text @ F.normalize(aerial[i], dim=0)
        self.assertTrue(torch.allclose(scores, expected, atol=1e-6))

    def test_uniform_mask_gives_plain_cosine(self):
        torch.manual_seed(1)
        text, aerial = torch.randn(4, 16), torch.randn(6, 16)
        plain = plain_cosine_scores(text, aerial)

        for value in (0.5, 1.0, 0.01):
            mask = torch.full((6, 16), value)
            self.assertTrue(
                torch.allclose(text_side_masked_scores(text, aerial, mask), plain, atol=1e-6)
            )

    def test_common_mask_scale_cancels(self):
        torch.manual_seed(2)
        text, aerial = torch.randn(4, 16), torch.randn(6, 16)
        mask = torch.sigmoid(torch.randn(6, 16))

        self.assertTrue(torch.allclose(
            text_side_masked_scores(text, aerial, mask),
            text_side_masked_scores(text, aerial, 0.37 * mask),
            atol=1e-6,
        ))

    def test_selective_mask_changes_scores(self):
        torch.manual_seed(3)
        text, aerial = torch.randn(4, 16), torch.randn(6, 16)
        mask = torch.sigmoid(4 * torch.randn(6, 16))

        self.assertFalse(torch.allclose(
            text_side_masked_scores(text, aerial, mask),
            plain_cosine_scores(text, aerial),
            atol=1e-4,
        ))

    def test_margin_zero_is_original_sdm_and_margin_raises_loss(self):
        torch.manual_seed(4)
        scores = torch.randn(6, 6) * 0.3 + torch.eye(6) * 0.5
        pids = torch.tensor([0, 0, 1, 2, 3, 3])
        scale = torch.tensor(50.0)

        original = compute_sdm_from_scores(scores, pids, scale)
        self.assertTrue(torch.equal(
            original, compute_sdm_from_scores(scores, pids, scale, margin=0.0)
        ))
        self.assertGreater(
            compute_sdm_from_scores(scores, pids, scale, margin=0.2).item(),
            original.item(),
        )

    def test_retrieval_scores_unpack_gallery_rows(self):
        torch.manual_seed(5)
        text, aerial = torch.randn(4, EMBED), torch.randn(3, EMBED)
        mask = torch.sigmoid(torch.randn(3, EMBED))

        scores = dpm_retrieval_scores(text, torch.cat([aerial, mask], dim=-1), EMBED)

        self.assertEqual(list(scores), ["plain", "masked", "sum"])
        self.assertTrue(torch.allclose(scores["plain"], plain_cosine_scores(text, aerial)))
        self.assertTrue(torch.allclose(
            scores["masked"], text_side_masked_scores(text, aerial, mask)
        ))
        self.assertTrue(torch.allclose(scores["sum"], scores["plain"] + scores["masked"]))
        with self.assertRaises(ValueError):
            dpm_retrieval_scores(text, aerial, EMBED)

    def test_mask_statistics(self):
        uniform = torch.full((3, 10), 0.7)
        instance_std, participation = mask_statistics(uniform)
        self.assertAlmostEqual(instance_std.item(), 0.0, places=6)
        self.assertAlmostEqual(participation.item(), 1.0, places=6)

        half_on = torch.cat([torch.ones(3, 5), torch.zeros(3, 5)], dim=1)
        _, participation = mask_statistics(half_on)
        self.assertAlmostEqual(participation.item(), 0.5, places=6)


class OcclusionAndIdentityUnitTest(unittest.TestCase):

    def test_occlude_band_blanks_one_contiguous_strip(self):
        torch.manual_seed(20)
        images = torch.ones(6, 3, 40, 8)
        occluded = occlude_band(images, 0.25)

        self.assertTrue(torch.equal(images, torch.ones(6, 3, 40, 8)))  # input untouched
        for image in occluded:
            blank_rows = (image == 0).all(dim=0).all(dim=1).nonzero().flatten()
            self.assertEqual(len(blank_rows), 10)
            self.assertEqual(int(blank_rows[-1] - blank_rows[0]), 9)
            self.assertEqual(int((image == 0).sum()), 10 * 3 * 8)

    def test_occlude_band_rejects_full_occlusion(self):
        with self.assertRaises(ValueError):
            occlude_band(torch.ones(2, 3, 10, 4), 1.0)

    def test_arcface_matches_dpm_definition(self):
        cosine = torch.tensor([[0.9, 0.2, -0.3], [0.1, 0.5, 0.4]])
        labels = torch.tensor([0, 2])

        plain = arcface_logits(cosine, labels, scale=30.0, margin=0.0)
        self.assertTrue(torch.allclose(plain, 30.0 * cosine, atol=1e-5))

        margin = arcface_logits(cosine, labels, scale=30.0, margin=0.5)
        expected_target = 30.0 * torch.cos(torch.acos(torch.tensor([0.9, 0.4])) + 0.5)
        self.assertTrue(torch.allclose(margin[[0, 1], [0, 2]], expected_target, atol=1e-4))
        off_target = torch.ones_like(cosine, dtype=torch.bool)
        off_target[[0, 1], [0, 2]] = False
        self.assertTrue(torch.allclose(margin[off_target], 30.0 * cosine[off_target]))

    def test_masked_prototype_cosine_matches_definition(self):
        torch.manual_seed(21)
        head = PrototypeHead(num_ids=5, embed_dim=16)
        aerial = torch.randn(3, 16)
        mask = torch.sigmoid(torch.randn(3, 16))

        cosine = head.masked_cosine(aerial, mask)
        self.assertEqual(cosine.shape, (3, 5))
        for i in range(3):
            for c in range(5):
                prototype = F.normalize(mask[i] * F.normalize(head.weight[c], dim=0), dim=0)
                self.assertAlmostEqual(
                    cosine[i, c].item(),
                    (prototype @ F.normalize(aerial[i], dim=0)).item(),
                    places=5,
                )

    def test_participation_ratio_is_differentiable(self):
        logits = torch.randn(2, 8, requires_grad=True)
        participation_ratio(torch.sigmoid(logits)).sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


def fake_hidden(batch, generator=None):
    return {
        layer: torch.randn(NUM_PATCHES + 1, batch, HIDDEN_WIDTH, generator=generator)
        for layer in HierarchicalMaskGenerator.LAYERS
    }


class MaskModuleTest(unittest.TestCase):

    def test_hmg_starts_at_half_and_learns(self):
        torch.manual_seed(6)
        hmg = HierarchicalMaskGenerator(HIDDEN_WIDTH, EMBED, GRID)
        mask = hmg(fake_hidden(3))

        self.assertEqual(mask.shape, (3, EMBED))
        self.assertTrue(torch.allclose(mask, torch.full_like(mask, 0.5)))

        (mask * torch.randn_like(mask)).sum().backward()
        self.assertGreater(hmg.fc.weight.grad.abs().sum().item(), 0.0)

    def test_hmg_rejects_wrong_grid(self):
        hmg = HierarchicalMaskGenerator(HIDDEN_WIDTH, EMBED, (12, 8))
        with self.assertRaises(ValueError):
            hmg(fake_hidden(2))

    def test_static_mask_is_shared(self):
        static = StaticMask(EMBED)
        mask = static(5)
        self.assertEqual(mask.shape, (5, EMBED))
        self.assertTrue(torch.allclose(mask, torch.full_like(mask, 0.5)))


class ForwardWithHiddenTest(unittest.TestCase):

    def test_matches_forward_in_train_and_eval(self):
        torch.manual_seed(7)
        vit = VisionTransformer(
            input_resolution=(64, 32), patch_size=16, stride_size=16,
            width=32, layers=4, heads=2, output_dim=16,
        )
        images = torch.randn(2, 3, 64, 32)

        for training in (True, False):
            vit.train(training)
            expected = vit(images)
            output, hidden = vit.forward_with_hidden(images, (1, 3))
            self.assertTrue(torch.allclose(output, expected, atol=1e-6))
            self.assertEqual(sorted(hidden), [1, 3])
            self.assertEqual(hidden[3].shape, (vit.num_x * vit.num_y + 1, 2, 32))


class DummyVisual(nn.Module):

    def __init__(self):
        super().__init__()
        self.transformer = SimpleNamespace(width=HIDDEN_WIDTH)
        self.num_y, self.num_x = GRID
        self.calls = 0

    def forward_with_hidden(self, image, layers):
        self.calls += 1
        batch = image.shape[0]
        hidden = {
            layer: image[:, :, :HIDDEN_WIDTH].permute(1, 0, 2) * (layer + 1)
            for layer in layers
        }
        return image, hidden


class DummyBaseModel(nn.Module):
    """Minimal CLIP replacement: images are already [B, N + 1, EMBED] tokens."""

    def __init__(self, text_feats):
        super().__init__()
        self.text_feats = text_feats
        self.visual = DummyVisual()
        self.dtype = torch.float32

    def forward(self, images, ground_images, caption_ids):
        return images, ground_images, self.text_feats

    def encode_image(self, image):
        return image

    def encode_text(self, text):
        return self.text_feats


def build_model(text_feats, base_model=None, **overrides):
    args = dict(
        loss_names="cda",
        pretrain_choice="dummy",
        img_size=(1, 1),
        stride_size=1,
        temperature=0.02,
        avm_mode="dpm",
        avm_loss_weight=1.0,
        avm_margin=0.0,
        avm_mask_input="cls",
        avm_mask_policy="learned",
        avm_detach_backbone=False,
        avm_eval_score="masked",
    )
    args.update(overrides)
    with patch(
        "model.build_finetune.build_CLIP_from_openai_pretrained",
        return_value=(base_model or DummyBaseModel(text_feats), {"embed_dim": EMBED}),
    ):
        return IRRA(SimpleNamespace(**args))


class DPMIntegrationTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(11)
        self.text_feats = torch.randn(4, 3, EMBED)
        self.images = torch.randn(4, NUM_PATCHES + 1, EMBED)
        self.batch = {
            "images": self.images,
            "ground_imgs": torch.randn(4, NUM_PATCHES + 1, EMBED),
            "caption_ids": torch.tensor([[1, 2, 9], [2, 3, 9], [3, 4, 9], [4, 5, 9]]),
            "pids": torch.tensor([0, 0, 1, 2]),
        }

    def run_forward(self, model):
        with no_autocast():
            return model(self.batch)

    def plain_sdm(self, margin):
        t_feats = self.text_feats[torch.arange(4), self.batch["caption_ids"].argmax(dim=-1)]
        return compute_sdm_from_scores(
            plain_cosine_scores(t_feats, self.images[:, 0, :]),
            self.batch["pids"],
            torch.ones([]) * 50.0,
            margin=margin,
        )

    def test_forward_keys_and_initial_mask_equals_plain_branch(self):
        model = build_model(self.text_feats, avm_margin=0.1)
        ret = self.run_forward(model)

        self.assertEqual(set(ret), {
            "cda_loss", "avm_ret_loss", "avm_mask_mean", "avm_mask_std",
            "avm_mask_inst_std", "avm_mask_eff",
        })
        self.assertAlmostEqual(ret["avm_mask_mean"].item(), 0.5, places=6)
        self.assertAlmostEqual(ret["avm_mask_inst_std"].item(), 0.0, places=6)
        self.assertAlmostEqual(ret["avm_mask_eff"].item(), 1.0, places=6)
        # Zero-initialised head -> uniform mask -> masked score = plain cosine.
        self.assertTrue(torch.allclose(ret["avm_ret_loss"], self.plain_sdm(0.1), atol=1e-5))

        (ret["cda_loss"] + ret["avm_ret_loss"]).backward()
        grad = model.avm_mask_head.mlp[-1].weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0.0)

    def test_margin_raises_masked_loss(self):
        loss_0 = self.run_forward(build_model(self.text_feats))["avm_ret_loss"]
        loss_2 = self.run_forward(build_model(self.text_feats, avm_margin=0.2))["avm_ret_loss"]
        self.assertGreater(loss_2.item(), loss_0.item())

    def test_detach_keeps_masked_loss_off_the_backbone(self):
        for detach in (False, True):
            images = self.images.clone().requires_grad_(True)
            self.batch["images"] = images
            model = build_model(self.text_feats, avm_detach_backbone=detach)
            with torch.no_grad():
                model.avm_mask_head.mlp[-1].bias.copy_(torch.linspace(-2, 2, EMBED))

            self.run_forward(model)["avm_ret_loss"].backward()

            if detach:
                self.assertIsNone(images.grad)
            else:
                self.assertGreater(images.grad.abs().sum().item(), 0.0)
            self.assertGreater(model.avm_mask_head.mlp[-1].weight.grad.abs().sum().item(), 0.0)

    def test_ones_policy_is_margin_only_control(self):
        model = build_model(self.text_feats, avm_mask_policy="ones", avm_margin=0.1)
        self.assertIsNone(model.avm_mask_head)
        ret = self.run_forward(model)
        self.assertAlmostEqual(ret["avm_mask_mean"].item(), 1.0, places=6)
        self.assertTrue(torch.allclose(ret["avm_ret_loss"], self.plain_sdm(0.1), atol=1e-5))

    def test_static_policy_shares_one_mask(self):
        model = build_model(self.text_feats, avm_mask_policy="static")
        self.assertIsInstance(model.avm_mask_head, StaticMask)
        with torch.no_grad():
            model.avm_mask_head.logits.copy_(torch.linspace(-2, 2, EMBED))
        packed = model.encode_image(self.images)
        mask = packed[:, EMBED:]
        self.assertTrue(torch.allclose(mask, mask[:1].expand_as(mask)))

    def test_encode_image_packs_cls_and_mask(self):
        model = build_model(self.text_feats)
        with torch.no_grad():
            model.avm_mask_head.mlp[-1].bias.copy_(torch.linspace(-2, 2, EMBED))

        packed = model.encode_image(self.images)
        aerial = self.images[:, 0, :]
        self.assertEqual(packed.shape, (4, 2 * EMBED))
        self.assertTrue(torch.allclose(packed[:, :EMBED], aerial))
        self.assertTrue(torch.allclose(packed[:, EMBED:], model.avm_mask_head(aerial)))

        text = torch.randn(3, EMBED)
        scores = model.retrieval_scores(text, packed)
        self.assertTrue(torch.allclose(
            scores["masked"], text_side_masked_scores(text, aerial, packed[:, EMBED:])
        ))

    def test_hmg_input_uses_hidden_states(self):
        model = build_model(self.text_feats, avm_mask_input="hmg", avm_margin=0.1)
        self.assertIsInstance(model.avm_mask_head, HierarchicalMaskGenerator)

        ret = self.run_forward(model)
        self.assertEqual(model.base_model.visual.calls, 1)
        (ret["cda_loss"] + ret["avm_ret_loss"]).backward()
        self.assertGreater(model.avm_mask_head.fc.weight.grad.abs().sum().item(), 0.0)

        model.eval()
        packed = model.encode_image(self.images)
        self.assertEqual(model.base_model.visual.calls, 2)
        self.assertEqual(packed.shape, (4, 2 * EMBED))

    def test_identity_losses_train_prototypes_and_mask(self):
        model = build_model(
            self.text_feats, avm_loss_weight=0.0,
            avm_id_plain_weight=0.5, avm_id_masked_weight=0.5, avm_id_classes=3,
        )
        ret = self.run_forward(model)

        self.assertIn("avm_id_loss", ret)
        self.assertIn("avm_mid_loss", ret)
        self.assertTrue(torch.isfinite(ret["avm_id_loss"]) and torch.isfinite(ret["avm_mid_loss"]))
        (ret["avm_id_loss"] + ret["avm_mid_loss"]).backward()
        self.assertGreater(model.avm_id_head.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.avm_mask_head.mlp[-1].weight.grad.abs().sum().item(), 0.0)

    def test_identity_pid_out_of_range_is_caught(self):
        model = build_model(self.text_feats, avm_id_plain_weight=0.5, avm_id_classes=2)
        with self.assertRaises(RuntimeError):
            self.run_forward(model)

    def test_negative_pid_needs_the_offset(self):
        # AERI-PEDES train pids are int(anno['pid']) - 1 and include -1;
        # batch 5 queue I died on a CUDA assert because of it.
        self.batch["pids"] = torch.tensor([-1, -1, 0, 1])
        with self.assertRaises(RuntimeError):
            self.run_forward(build_model(
                self.text_feats, avm_id_plain_weight=0.5, avm_id_masked_weight=0.5, avm_id_classes=3,
            ))

        model = build_model(
            self.text_feats, avm_id_plain_weight=0.5, avm_id_masked_weight=0.5,
            avm_id_classes=3, avm_id_offset=1,
        )
        ret = self.run_forward(model)
        self.assertTrue(torch.isfinite(ret["avm_id_loss"]) and torch.isfinite(ret["avm_mid_loss"]))

    def test_eff_floor_only_bites_below_the_floor(self):
        model = build_model(self.text_feats, avm_margin=0.1, avm_eff_floor=0.9, avm_eff_floor_weight=10.0)

        uniform = self.run_forward(model)
        self.assertAlmostEqual(uniform["avm_eff_floor_loss"].item(), 0.0, places=6)

        with torch.no_grad():
            model.avm_mask_head.mlp[-1].bias.copy_(torch.linspace(-4, 4, EMBED))
        selective = self.run_forward(model)
        self.assertLess(selective["avm_mask_eff"].item(), 0.9)
        self.assertAlmostEqual(
            selective["avm_eff_floor_loss"].item(),
            10.0 * (0.9 - selective["avm_mask_eff"].item()),
            places=4,
        )
        selective["avm_eff_floor_loss"].backward()
        self.assertGreater(model.avm_mask_head.mlp[-1].bias.grad.abs().sum().item(), 0.0)

    def test_occlusion_losses_only_train_the_mask(self):
        images = self.images.clone().requires_grad_(True)
        self.batch["images"] = images
        model = build_model(
            self.text_feats, avm_margin=0.1,
            avm_occ_ratio=0.25, avm_occ_weight=1.0, avm_occ_rank_weight=1.0,
        )
        with torch.no_grad():
            model.avm_mask_head.mlp[-1].bias.copy_(torch.linspace(-2, 2, EMBED))

        ret = self.run_forward(model)
        for key in ("avm_occ_loss", "avm_occ_rank_loss", "avm_occ_eff"):
            self.assertIn(key, ret)
        self.assertTrue(torch.isfinite(ret["avm_occ_loss"]))

        (ret["avm_occ_loss"] + ret["avm_occ_rank_loss"]).backward()
        self.assertIsNone(images.grad)
        self.assertGreater(model.avm_mask_head.mlp[-1].weight.grad.abs().sum().item(), 0.0)

    def test_occlusion_with_hmg_encodes_the_occluded_copy(self):
        model = build_model(self.text_feats, avm_mask_input="hmg", avm_occ_ratio=0.5)
        ret = self.run_forward(model)
        self.assertEqual(model.base_model.visual.calls, 2)
        ret["avm_occ_loss"].backward()
        self.assertGreater(model.avm_mask_head.fc.weight.grad.abs().sum().item(), 0.0)

    # ---------------------------------------------------------------- batch 7

    def test_batch7_switches_default_off(self):
        model = build_model(
            self.text_feats, avm_margin=0.1, avm_mask_groups=0,
            avm_occ_vis_weight=0.0, avm_eval_perm=False, avm_two_step=False,
        )
        ret = self.run_forward(model)
        self.assertEqual(set(ret), {
            "cda_loss", "avm_ret_loss", "avm_mask_mean", "avm_mask_std",
            "avm_mask_inst_std", "avm_mask_eff",
        })
        scores = model.retrieval_scores(torch.randn(3, EMBED), model.encode_image(self.images))
        self.assertEqual(list(scores), ["plain", "masked", "sum"])

    def test_grouped_mask_in_the_model(self):
        model = build_model(self.text_feats, avm_margin=0.1, avm_mask_groups=4)
        self.assertEqual(model.avm_mask_head.mlp[-1].out_features, 4)
        ret = self.run_forward(model)
        # Still starts as the plain branch.
        self.assertTrue(torch.allclose(ret["avm_ret_loss"], self.plain_sdm(0.1), atol=1e-5))
        (ret["cda_loss"] + ret["avm_ret_loss"]).backward()
        self.assertGreater(model.avm_mask_head.mlp[-1].weight.grad.abs().sum().item(), 0.0)

        with torch.no_grad():
            model.avm_mask_head.mlp[-1].bias.copy_(torch.linspace(-2, 2, 4))
        blocks = model.encode_image(self.images)[:, EMBED:].view(4, 4, 2)
        self.assertTrue(torch.equal(blocks[..., 0], blocks[..., 1]))

        hmg = build_model(self.text_feats, avm_mask_input="hmg", avm_mask_groups=2)
        self.assertEqual(hmg.avm_mask_head.fc.out_features, 2)
        self.assertEqual(hmg.encode_image(self.images).shape, (4, 2 * EMBED))

    def test_eval_perm_adds_the_control_rows(self):
        model = build_model(self.text_feats, avm_eval_perm=True)
        scores = model.retrieval_scores(torch.randn(3, EMBED), model.encode_image(self.images))
        self.assertEqual(list(scores), ["plain", "masked", "sum", "masked_perm", "sum_perm"])

    def partial_occlusion(self):
        # Deterministic stand-in for occlude_band: every image loses half of
        # its own CLS token. (Swapping images across the batch would pair a
        # symmetric target with an antisymmetric mask difference, and the
        # correlations of each pair would cancel.)
        def blank_half_cls(images, ratio):
            occluded = images.clone()
            occluded[:, 0, : EMBED // 2] = 0
            return occluded
        return patch("model.build_finetune.occlude_band", side_effect=blank_half_cls)

    def test_visibility_target_is_pid_free_and_only_trains_the_mask(self):
        images = self.images.clone().requires_grad_(True)
        self.batch["images"] = images
        model = build_model(
            self.text_feats, avm_margin=0.1,
            avm_occ_ratio=0.25, avm_occ_weight=0.0, avm_occ_vis_weight=1.0,
        )
        with self.partial_occlusion():
            ret = self.run_forward(model)
            self.batch["pids"] = torch.tensor([3, 2, 1, 0])
            other_pids = self.run_forward(model)

        self.assertIn("avm_occ_vis_loss", ret)
        self.assertNotIn("avm_occ_loss", ret)
        self.assertNotIn("avm_occ_rank_loss", ret)
        self.assertTrue(torch.allclose(ret["avm_occ_vis_loss"], other_pids["avm_occ_vis_loss"]))
        # Both masks start at exactly 0.5: correlation 0, loss 1, finite gradient.
        self.assertAlmostEqual(ret["avm_occ_vis_loss"].item(), 1.0, places=5)

        ret["avm_occ_vis_loss"].backward()
        self.assertIsNone(images.grad)
        grad = model.avm_mask_head.mlp[-1].weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0.0)

    def test_visibility_loss_is_learnable_by_the_mask_alone(self):
        model = build_model(
            self.text_feats, avm_occ_ratio=0.25, avm_occ_weight=0.0, avm_occ_vis_weight=1.0,
        )
        optimizer = torch.optim.Adam(model.avm_mask_head.parameters(), lr=1e-2)
        with self.partial_occlusion():
            first = self.run_forward(model)["avm_occ_vis_loss"].item()
            for _ in range(40):
                loss = self.run_forward(model)["avm_occ_vis_loss"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            last = self.run_forward(model)["avm_occ_vis_loss"].item()
        self.assertLess(last, first - 0.2)

    def test_invalid_configurations_are_rejected(self):
        bad = [
            dict(avm_mode="feature", avm_margin=0.1),
            dict(avm_mode="none", avm_mask_policy="static"),
            dict(avm_mask_policy="static", avm_mask_input="hmg"),
            dict(avm_mask_policy="ones", avm_detach_backbone=True),
            dict(avm_margin=-0.1),
            dict(avm_mode="feature", avm_occ_ratio=0.25),
            dict(avm_mode="none", avm_id_plain_weight=0.5, avm_id_classes=3),
            dict(avm_occ_ratio=0.25, avm_mask_policy="static"),
            dict(avm_occ_ratio=1.0),
            dict(avm_occ_ratio=0.25, avm_occ_weight=0.0, avm_occ_rank_weight=0.0),
            dict(avm_id_masked_weight=0.5, avm_id_classes=0),
            dict(avm_mode="feature", avm_eff_floor=0.9),
            dict(avm_mask_policy="ones", avm_eff_floor=0.9),
            dict(avm_eff_floor=1.0),
            dict(avm_eff_floor=0.9, avm_eff_floor_weight=0.0),
            # Batch 7.
            dict(avm_mode="feature", avm_mask_groups=4),
            dict(avm_mode="none", avm_eval_perm=True),
            dict(avm_mode="none", avm_two_step=True),
            dict(avm_mode="none", avm_occ_vis_weight=1.0),
            dict(avm_mask_groups=3),
            dict(avm_mask_groups=EMBED),
            dict(avm_mask_groups=4, avm_mask_policy="static"),
            dict(avm_occ_vis_weight=1.0),
            dict(avm_occ_ratio=0.25, avm_occ_weight=0.0, avm_occ_vis_weight=-1.0),
            dict(avm_mask_policy="ones", avm_two_step=True),
        ]
        for overrides in bad:
            with self.subTest(**overrides), self.assertRaises(ValueError):
                build_model(self.text_feats, **overrides)


class Batch7UnitTest(unittest.TestCase):

    def test_expand_groups_tiles_contiguous_blocks(self):
        torch.manual_seed(30)
        gates = torch.sigmoid(torch.randn(3, 4))
        mask = expand_groups(gates, 8)
        self.assertEqual(mask.shape, (3, 8))
        for g in range(4):
            self.assertTrue(torch.equal(mask[:, 2 * g], gates[:, g]))
            self.assertTrue(torch.equal(mask[:, 2 * g + 1], gates[:, g]))
        # The eff floor keeps its meaning on the expanded mask.
        self.assertTrue(torch.allclose(participation_ratio(mask), participation_ratio(gates)))

    def test_group_count_must_tile_the_channels(self):
        check_mask_groups(8, 0)
        check_mask_groups(8, 4)
        for groups in (1, 3, 8, 16):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                check_mask_groups(8, groups)

    def test_grouped_heads_start_at_half_and_share_gates(self):
        torch.manual_seed(31)
        cls_head = FeatureMaskHead(EMBED, groups=4)
        hmg = HierarchicalMaskGenerator(HIDDEN_WIDTH, EMBED, GRID, groups=2)
        self.assertEqual(cls_head.mlp[-1].out_features, 4)
        self.assertEqual(hmg.fc.out_features, 2)
        for mask in (cls_head(torch.randn(3, EMBED)), hmg(fake_hidden(3))):
            self.assertEqual(mask.shape, (3, EMBED))
            self.assertTrue(torch.allclose(mask, torch.full_like(mask, 0.5)))

        with torch.no_grad():
            cls_head.mlp[-1].bias.copy_(torch.tensor([-2.0, -1.0, 1.0, 2.0]))
        mask = cls_head(torch.randn(3, EMBED))
        blocks = mask.view(3, 4, 2)
        self.assertTrue(torch.equal(blocks[..., 0], blocks[..., 1]))
        self.assertLess(participation_ratio(mask).mean().item(), 1.0)

    def test_fixed_derangement(self):
        for n in (2, 5, 100):
            perm = fixed_derangement(n)
            self.assertTrue(torch.equal(perm.sort().values, torch.arange(n)))
            self.assertTrue((perm != torch.arange(n)).all())
            self.assertTrue(torch.equal(perm, fixed_derangement(n)))
        with self.assertRaises(ValueError):
            fixed_derangement(1)

    def test_permuted_rows_swap_masks_between_images(self):
        torch.manual_seed(32)
        text, aerial = torch.randn(4, EMBED), torch.randn(5, EMBED)
        mask = torch.sigmoid(2 * torch.randn(5, EMBED))
        scores = dpm_retrieval_scores(text, torch.cat([aerial, mask], dim=-1), EMBED, permute=True)

        self.assertEqual(list(scores), ["plain", "masked", "sum", "masked_perm", "sum_perm"])
        shuffled = mask[fixed_derangement(5)]
        self.assertTrue(torch.allclose(
            scores["masked_perm"], text_side_masked_scores(text, aerial, shuffled)
        ))
        self.assertTrue(torch.allclose(scores["sum_perm"], scores["plain"] + scores["masked_perm"]))
        self.assertFalse(torch.allclose(scores["masked_perm"], scores["masked"], atol=1e-4))

        # One mask shared by every image: the permutation has nothing to break.
        shared = mask[:1].expand_as(mask)
        same = dpm_retrieval_scores(text, torch.cat([aerial, shared], dim=-1), EMBED, permute=True)
        self.assertTrue(torch.allclose(same["masked_perm"], same["masked"]))

    def test_soft_row_correlation(self):
        torch.manual_seed(33)
        target = torch.rand(3, 64)
        x = 5.0 * torch.randn(3, 64)
        xc, tc = x - x.mean(1, keepdim=True), target - target.mean(1, keepdim=True)
        pearson = (xc * tc).sum(1) / (xc.norm(dim=1) * tc.norm(dim=1))
        self.assertTrue(torch.allclose(soft_row_correlation(x, target), pearson, atol=1e-4))
        self.assertTrue(torch.allclose(soft_row_correlation(-x, target), -pearson, atol=1e-4))

        flat = torch.zeros(3, 64, requires_grad=True)
        corr = soft_row_correlation(flat, target)
        self.assertTrue(torch.allclose(corr, torch.zeros(3)))
        corr.sum().backward()
        self.assertTrue(torch.isfinite(flat.grad).all())
        self.assertGreater(flat.grad.abs().sum().item(), 0.0)
        self.assertLess(flat.grad.abs().max().item(), 10.0)


class TrainableDummyBaseModel(DummyBaseModel):
    """DummyBaseModel with a backbone parameter for the main optimizer."""

    def __init__(self, text_feats):
        super().__init__(text_feats)
        self.shift = nn.Parameter(torch.zeros(EMBED))

    def forward(self, images, ground_images, caption_ids):
        return images + self.shift, ground_images + self.shift, self.text_feats

    def encode_image(self, image):
        return image + self.shift


class TwoStepUpdateTest(unittest.TestCase):
    """--avm_two_step: DPM's per-iteration two-step update."""

    def setUp(self):
        torch.manual_seed(40)
        text_feats = torch.randn(4, 3, EMBED)
        self.batch = {
            "images": torch.randn(4, NUM_PATCHES + 1, EMBED),
            "ground_imgs": torch.randn(4, NUM_PATCHES + 1, EMBED),
            "caption_ids": torch.tensor([[1, 2, 9], [2, 3, 9], [3, 4, 9], [4, 5, 9]]),
            "pids": torch.tensor([0, 0, 1, 2]),
        }
        self.model = build_model(
            text_feats, base_model=TrainableDummyBaseModel(text_feats),
            avm_margin=0.1, avm_two_step=True,
        )
        with torch.no_grad():
            self.model.avm_mask_head.mlp[-1].bias.copy_(torch.linspace(-2, 2, EMBED))
        self.args = SimpleNamespace(
            optimizer="Adam", lr=1e-3, lr2=1e-3, lr_factor=5.0, avm_lr=1e-2,
            weight_decay=0.0, alpha=0.9, beta=0.999, momentum=0.9, avm_two_step=True,
        )

    def mask_names(self):
        return {n for n, _ in self.model.named_parameters() if "avm_mask_head" in n}

    def snapshot(self):
        return {n: p.detach().clone() for n, p in self.model.named_parameters()}

    def changed(self, before):
        return {n for n, p in self.model.named_parameters() if not torch.equal(p, before[n])}

    def test_optimizers_split_the_parameters(self):
        main = build_optimizer(self.args, self.model)
        mask = build_mask_optimizer(self.args, self.model)
        in_main = {id(p) for g in main.param_groups for p in g["params"]}
        in_mask = {id(p) for g in mask.param_groups for p in g["params"]}
        mask_ids = {id(p) for n, p in self.model.named_parameters() if "avm_mask_head" in n}

        self.assertEqual(in_mask, mask_ids)
        self.assertFalse(in_main & mask_ids)
        self.assertIn(id(self.model.base_model.shift), in_main)
        self.assertTrue(all(g["lr"] == 1e-2 for g in mask.param_groups))

        one_step = SimpleNamespace(**{**vars(self.args), "avm_two_step": False})
        in_one = {id(p) for g in build_optimizer(one_step, self.model).param_groups for p in g["params"]}
        self.assertTrue(mask_ids <= in_one)

    def test_step_one_skips_the_mask_and_step_two_moves_only_the_mask(self):
        main = build_optimizer(self.args, self.model)
        mask = build_mask_optimizer(self.args, self.model)

        before = self.snapshot()
        with no_autocast():
            _, loss = forward_losses(self.model, self.batch)
        main.zero_grad()
        loss.backward()
        main.step()
        step_one = self.changed(before)
        self.assertIn("base_model.shift", step_one)
        self.assertFalse(step_one & self.mask_names())

        before = self.snapshot()
        with no_autocast():
            mask_generator_step(self.model, self.batch, mask)
        step_two = self.changed(before)
        self.assertTrue(step_two)
        self.assertTrue(step_two <= self.mask_names())

    def test_mask_optimizer_needs_a_mask_generator(self):
        model = build_model(torch.randn(4, 3, EMBED), avm_mode="none")
        with self.assertRaises(ValueError):
            build_mask_optimizer(self.args, model)


class EvaluatorNamedScoresTest(unittest.TestCase):

    def test_exactly_one_t2i_row_and_selected_rsum(self):
        torch.manual_seed(12)
        qids = torch.arange(12)
        gids = torch.arange(12)
        good = torch.eye(12) + 0.01 * torch.randn(12, 12)
        bad = torch.randn(12, 12)

        model = SimpleNamespace(
            avm_mode="dpm",
            avm_eval_score="sum",
            retrieval_scores=lambda q, g: {"plain": bad, "masked": bad, "sum": good},
        )
        evaluator = Evaluator(img_loader=None, txt_loader=None)

        with self.assertLogs("IRRA.eval", level=logging.INFO) as logs:
            rsum = evaluator._eval_named_scores(model, None, None, qids, gids)

        self.assertAlmostEqual(float(rsum), 300.0, places=4)
        text = "\n".join(logs.output)
        t2i_rows = [line for line in text.splitlines() if re.match(r"^\|\s*t2i", line)]
        self.assertEqual(len(t2i_rows), 1)
        for name in ("plain", "masked", "sum"):
            self.assertRegex(text, rf"\|\s*{name}\s*\|")

    def test_permuted_rows_do_not_collide_with_report_names(self):
        # report.py v3 reads rows with ^|\s*(t2i|plain|masked|sum)\s*| and
        # must keep reading one row per name when the perm rows are added.
        torch.manual_seed(13)
        qids = gids = torch.arange(12)
        scores = {name: torch.randn(12, 12) for name in
                  ("plain", "masked", "sum", "masked_perm", "sum_perm")}
        model = SimpleNamespace(
            avm_mode="dpm", avm_eval_score="masked",
            retrieval_scores=lambda q, g: scores,
        )
        with self.assertLogs("IRRA.eval", level=logging.INFO) as logs:
            Evaluator(img_loader=None, txt_loader=None)._eval_named_scores(model, None, None, qids, gids)

        rows = "\n".join(logs.output).splitlines()
        v3 = [re.match(r"^\|\s*(t2i|plain|masked|sum)\s*\|", line) for line in rows]
        self.assertEqual(sorted(m.group(1) for m in v3 if m), ["masked", "plain", "sum", "t2i"])
        for name in ("masked_perm", "sum_perm"):
            self.assertEqual(sum(bool(re.match(rf"^\|\s*{name}\s*\|", line)) for line in rows), 1)


if __name__ == "__main__":
    unittest.main()
