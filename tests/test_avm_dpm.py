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
    HierarchicalMaskGenerator,
    PrototypeHead,
    StaticMask,
    arcface_logits,
    dpm_retrieval_scores,
    mask_statistics,
    occlude_band,
    participation_ratio,
    plain_cosine_scores,
    text_side_masked_scores,
)
from model.build_finetune import IRRA
from model.clip_model import VisionTransformer
from model.objectives import compute_sdm_from_scores
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


def build_model(text_feats, **overrides):
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
        return_value=(DummyBaseModel(text_feats), {"embed_dim": EMBED}),
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
        ]
        for overrides in bad:
            with self.subTest(**overrides), self.assertRaises(ValueError):
                build_model(self.text_feats, **overrides)


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


if __name__ == "__main__":
    unittest.main()
