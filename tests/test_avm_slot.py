import unittest

import torch
import torch.nn.functional as F

from model.avm import (
    SemanticSlotPool,
    SlotMaskHead,
    batched_slot_dot,
    slot_gallery_embedding,
    slot_scores,
)
from model.objectives import compute_sdm, compute_sdm_from_scores
from utils.metrics import compute_retrieval_similarity


class SemanticSlotTest(unittest.TestCase):

    def test_slot_pool_shape_and_padding_mask(self):
        torch.manual_seed(1)

        tokens = torch.randn(2, 5, 8)
        valid_mask = torch.tensor([
            [True, True, True, False, False],
            [True, True, True, True, False],
        ])
        altered_tokens = tokens.clone()
        altered_tokens[~valid_mask] = 1000.0

        pool = SemanticSlotPool(embed_dim=8, num_slots=8)
        slots_a = pool(tokens, valid_mask=valid_mask)
        slots_b = pool(altered_tokens, valid_mask=valid_mask)

        self.assertEqual(slots_a.shape, (2, 8, 8))
        self.assertTrue(
            torch.allclose(slots_a, slots_b, atol=1e-6)
        )

    def test_initial_slot_mask_is_half(self):
        torch.manual_seed(2)

        head = SlotMaskHead(embed_dim=16, num_slots=8)
        mask = head(torch.randn(3, 16))

        self.assertEqual(mask.shape, (3, 8))
        self.assertTrue(
            torch.allclose(mask, torch.full_like(mask, 0.5))
        )

    def test_gallery_slot_norm_retains_mask(self):
        torch.manual_seed(3)

        aerial_slots = torch.randn(3, 8, 16)
        mask = torch.sigmoid(torch.randn(3, 8))
        gallery = slot_gallery_embedding(aerial_slots, mask)

        self.assertTrue(
            torch.allclose(
                gallery.norm(dim=-1), mask, atol=1e-6
            )
        )

    def test_slot_scores_match_weighted_cosine_sum(self):
        torch.manual_seed(4)

        text_slots = torch.randn(2, 8, 16)
        aerial_slots = torch.randn(3, 8, 16)
        mask = torch.sigmoid(torch.randn(3, 8))

        actual = slot_scores(text_slots, aerial_slots, mask)
        text_unit = F.normalize(text_slots, p=2, dim=-1)
        aerial_unit = F.normalize(aerial_slots, p=2, dim=-1)
        expected = (
            torch.einsum(
                "qkd,gkd->qgk", text_unit, aerial_unit
            )
            * mask.unsqueeze(0)
        ).sum(dim=-1)

        self.assertTrue(
            torch.allclose(actual, expected, atol=1e-6)
        )

    def test_slot_loss_reaches_pool_and_mask_head(self):
        torch.manual_seed(5)

        pool = SemanticSlotPool(embed_dim=8, num_slots=8)
        head = SlotMaskHead(embed_dim=8, num_slots=8)
        text_slots = pool(torch.randn(4, 5, 8))
        aerial_slots = pool(torch.randn(4, 6, 8))
        mask = head(torch.randn(4, 8))
        scores = slot_scores(text_slots, aerial_slots, mask)

        loss = compute_sdm_from_scores(
            raw_scores_t2i=scores,
            pid=torch.arange(4),
            logit_scale=torch.tensor(1.0),
        )
        loss.backward()

        query_grad = pool.slot_queries.grad
        mask_grad = head.mlp[-1].weight.grad

        self.assertIsNotNone(query_grad)
        self.assertIsNotNone(mask_grad)
        self.assertTrue(torch.isfinite(query_grad).all())
        self.assertTrue(torch.isfinite(mask_grad).all())
        self.assertGreater(query_grad.abs().sum().item(), 0.0)
        self.assertGreater(mask_grad.abs().sum().item(), 0.0)

    def test_score_sdm_matches_original_sdm(self):
        torch.manual_seed(8)

        image_global = torch.randn(4, 16)
        text_global = torch.randn(4, 16)
        pid = torch.arange(4)
        logit_scale = torch.tensor(10.0)
        raw_scores = (
            F.normalize(text_global, p=2, dim=-1)
            @ F.normalize(image_global, p=2, dim=-1).t()
        )

        expected = compute_sdm(
            image_global,
            text_global,
            pid,
            logit_scale,
        )
        actual = compute_sdm_from_scores(
            raw_scores,
            pid,
            logit_scale,
        )

        self.assertTrue(
            torch.allclose(actual, expected, atol=1e-6)
        )

    def test_score_sdm_supports_multi_positive_pids(self):
        torch.manual_seed(9)

        raw_scores = torch.randn(4, 4, requires_grad=True)
        pid = torch.tensor([0, 0, 1, 1])
        loss = compute_sdm_from_scores(
            raw_scores,
            pid,
            torch.tensor(10.0),
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(raw_scores.grad)
        self.assertTrue(torch.isfinite(raw_scores.grad).all())
        self.assertGreater(raw_scores.grad.abs().sum().item(), 0.0)

    def test_evaluator_similarity_uses_masked_slot_dot(self):
        torch.manual_seed(6)

        text_slots = torch.randn(2, 8, 16)
        aerial_slots = torch.randn(3, 8, 16)
        mask = torch.sigmoid(torch.randn(3, 8))
        gallery = slot_gallery_embedding(aerial_slots, mask)

        expected = batched_slot_dot(text_slots, gallery)
        actual = compute_retrieval_similarity(
            text_slots, gallery
        )

        self.assertTrue(
            torch.allclose(actual, expected, atol=1e-6)
        )

    def test_global_evaluator_similarity_is_unchanged(self):
        torch.manual_seed(7)

        text_global = torch.randn(2, 16)
        image_global = torch.randn(3, 16)
        expected = (
            F.normalize(text_global, p=2, dim=-1)
            @ F.normalize(image_global, p=2, dim=-1).t()
        )
        actual = compute_retrieval_similarity(
            text_global, image_global
        )

        self.assertTrue(
            torch.allclose(actual, expected, atol=1e-6)
        )


if __name__ == "__main__":
    unittest.main()
