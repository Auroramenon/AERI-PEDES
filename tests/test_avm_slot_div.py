import unittest

import torch

from model.objectives import compute_slot_decorrelation_loss


class SlotDecorrelationTest(unittest.TestCase):

    def test_identical_slots_have_unit_penalty(self):
        base = torch.randn(2, 1, 8)
        slots = base.expand(-1, 4, -1).clone()

        loss = compute_slot_decorrelation_loss(slots)

        self.assertAlmostEqual(loss.item(), 1.0, places=6)

    def test_orthogonal_slots_have_zero_penalty(self):
        slots = torch.eye(4).unsqueeze(0)

        loss = compute_slot_decorrelation_loss(slots)

        self.assertAlmostEqual(loss.item(), 0.0, places=7)

    def test_loss_reaches_slot_tensor(self):
        torch.manual_seed(21)
        slots = torch.randn(3, 4, 8, requires_grad=True)

        loss = compute_slot_decorrelation_loss(slots)
        loss.backward()

        self.assertIsNotNone(slots.grad)
        self.assertTrue(torch.isfinite(slots.grad).all())
        self.assertGreater(slots.grad.abs().sum().item(), 0.0)

    def test_single_slot_returns_differentiable_zero(self):
        slots = torch.randn(2, 1, 8, requires_grad=True)

        loss = compute_slot_decorrelation_loss(slots)
        loss.backward()

        self.assertEqual(loss.item(), 0.0)
        self.assertIsNotNone(slots.grad)
        self.assertTrue(torch.equal(
            slots.grad, torch.zeros_like(slots.grad)
        ))

    def test_invalid_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "\[B, K, D\]"):
            compute_slot_decorrelation_loss(torch.randn(3, 8))


if __name__ == "__main__":
    unittest.main()
