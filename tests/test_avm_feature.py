import unittest

import torch
import torch.nn.functional as F

from model.avm import (
    FeatureMaskHead,
    feature_gallery_embedding,
    feature_scores,
)
from model.objectives import compute_sdm_from_scores


class FeatureMaskTest(unittest.TestCase):

    def test_initial_mask_is_half(self):
        torch.manual_seed(1)

        aerial = torch.randn(3, 16)
        head = FeatureMaskHead(embed_dim=16)
        mask = head(aerial)

        self.assertEqual(mask.shape, (3, 16))
        self.assertTrue(
            torch.allclose(mask, torch.full_like(mask, 0.5))
        )

    def test_initial_gallery_matches_baseline(self):
        torch.manual_seed(2)

        aerial = torch.randn(3, 16)
        head = FeatureMaskHead(embed_dim=16)
        mask = head(aerial)

        masked_gallery = feature_gallery_embedding(aerial, mask)
        baseline_gallery = F.normalize(
            aerial.float(), p=2, dim=-1
        )

        self.assertTrue(
            torch.allclose(
                masked_gallery,
                baseline_gallery,
                atol=1e-6,
            )
        )

    def test_common_mask_scale_is_removed(self):
        torch.manual_seed(3)

        aerial = torch.randn(3, 16)
        mask = torch.sigmoid(torch.randn(3, 16))

        gallery_a = feature_gallery_embedding(aerial, mask)
        gallery_b = feature_gallery_embedding(
            aerial, mask * 0.37
        )

        self.assertTrue(
            torch.allclose(gallery_a, gallery_b, atol=1e-6)
        )

    def test_retrieval_loss_reaches_mask_head(self):
        torch.manual_seed(7)

        text = torch.randn(4, 8)
        aerial = torch.randn(4, 8)
        pids = torch.arange(4)

        head = FeatureMaskHead(embed_dim=8)
        mask = head(aerial)
        raw_scores = feature_scores(text, aerial, mask)

        self.assertEqual(raw_scores.shape, (4, 4))

        loss = compute_sdm_from_scores(
            raw_scores_t2i=raw_scores,
            pid=pids,
            logit_scale=torch.tensor(10.0),
        )
        loss.backward()

        final_weight_grad = head.mlp[-1].weight.grad

        self.assertIsNotNone(final_weight_grad)
        self.assertTrue(torch.isfinite(final_weight_grad).all())
        self.assertGreater(
            final_weight_grad.abs().sum().item(), 0.0
        )


if __name__ == "__main__":
    unittest.main()
