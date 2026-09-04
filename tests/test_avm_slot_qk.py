import unittest

import torch
import torch.nn.functional as F

from model.avm import ground_aerial_slot_target


class GroundAerialSlotTargetTest(unittest.TestCase):

    def test_qk_matches_guide_formula(self):
        torch.manual_seed(21)
        ground_slots = torch.randn(3, 8, 16)
        aerial_slots = torch.randn(3, 8, 16)
        temperature = 0.4

        actual = ground_aerial_slot_target(
            ground_slots, aerial_slots, temperature
        )
        expected = torch.sigmoid(
            F.cosine_similarity(
                ground_slots.float(),
                aerial_slots.float(),
                dim=-1,
            )
            / temperature
        ).detach()

        self.assertEqual(actual.shape, (3, 8))
        self.assertTrue(
            torch.allclose(actual, expected, atol=1e-7)
        )
        self.assertTrue(torch.all(actual >= 0))
        self.assertTrue(torch.all(actual <= 1))

    def test_qk_is_detached(self):
        torch.manual_seed(22)
        ground_slots = torch.randn(
            2, 8, 16, requires_grad=True
        )
        aerial_slots = torch.randn(
            2, 8, 16, requires_grad=True
        )

        target = ground_aerial_slot_target(
            ground_slots, aerial_slots, temperature=0.5
        )

        self.assertFalse(target.requires_grad)
        self.assertIsNone(target.grad_fn)

    def test_qk_rejects_mismatched_shapes(self):
        with self.assertRaisesRegex(
            ValueError, "identical shapes"
        ):
            ground_aerial_slot_target(
                torch.randn(2, 8, 16),
                torch.randn(2, 6, 16),
                temperature=0.5,
            )

    def test_qk_rejects_non_positive_temperature(self):
        ground_slots = torch.randn(2, 8, 16)
        aerial_slots = torch.randn(2, 8, 16)

        for temperature in (None, 0.0, -0.1):
            with self.subTest(temperature=temperature):
                with self.assertRaisesRegex(
                    ValueError, "temperature must be positive"
                ):
                    ground_aerial_slot_target(
                        ground_slots,
                        aerial_slots,
                        temperature,
                    )


if __name__ == "__main__":
    unittest.main()
