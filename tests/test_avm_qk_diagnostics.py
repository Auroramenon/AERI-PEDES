import unittest

import torch

from tools.diagnose_avm_qk import (
    describe,
    make_pid_mismatch_indices,
    normalized_attention_entropy,
    off_diagonal_values,
    pairwise_slot_cosine,
)


class QKDiagnosticTest(unittest.TestCase):
    def test_pid_mismatch_indices_always_change_pid(self):
        pids = torch.tensor([10, 10, 20, 30])
        indices = make_pid_mismatch_indices(pids)

        self.assertEqual(indices.shape, pids.shape)
        self.assertTrue(torch.all(pids[indices] != pids))

    def test_pid_mismatch_rejects_single_pid_batch(self):
        with self.assertRaisesRegex(ValueError, "different PID"):
            make_pid_mismatch_indices(torch.tensor([7, 7, 7]))

    def test_pairwise_slot_cosine_has_unit_diagonal(self):
        slots = torch.randn(3, 4, 6)
        similarity = pairwise_slot_cosine(slots)
        diagonal = torch.diagonal(similarity, dim1=1, dim2=2)

        self.assertEqual(similarity.shape, (3, 4, 4))
        self.assertTrue(
            torch.allclose(diagonal, torch.ones_like(diagonal))
        )

    def test_off_diagonal_values_exclude_diagonal(self):
        similarity = torch.tensor(
            [[[1.0, 2.0, 3.0], [4.0, 1.0, 5.0], [6.0, 7.0, 1.0]]]
        )
        values = off_diagonal_values(similarity)

        self.assertEqual(values.shape, (1, 6))
        self.assertEqual(
            sorted(values.flatten().tolist()),
            [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        )

    def test_uniform_attention_has_unit_normalized_entropy(self):
        attention = torch.full((2, 3, 5), 0.2)
        entropy = normalized_attention_entropy(attention)

        self.assertTrue(
            torch.allclose(entropy, torch.ones_like(entropy))
        )

    def test_describe_reports_population_statistics(self):
        statistics = describe(torch.tensor([1.0, 2.0, 3.0]))

        self.assertEqual(statistics["count"], 3)
        self.assertAlmostEqual(statistics["mean"], 2.0)
        self.assertAlmostEqual(
            statistics["std"], (2.0 / 3.0) ** 0.5
        )


if __name__ == "__main__":
    unittest.main()
