import unittest

import torch

from segment.lcds import LCDSBoundaryHead, LocalDependencyShift


class LCDSTest(unittest.TestCase):
    def test_dependency_drop_responds_to_local_minimum(self):
        head = LocalDependencyShift(4, window_size=2, radius=1)
        result = head.dependency_drop(torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0]))
        self.assertEqual(result.argmax().item(), 2)
        self.assertEqual(result.max().item(), 1.0)

    def test_candidate_positions_and_no_padding_boundaries(self):
        torch.manual_seed(42)
        head = LocalDependencyShift(4, window_size=2)
        shift, loss = head(torch.randn(9, 4))
        self.assertEqual(shift.shape, (9,))
        self.assertEqual(shift[0].item(), 0.0)
        self.assertTrue(torch.equal(shift[-2:], torch.zeros(2)))
        self.assertTrue(torch.isfinite(loss))
        (loss + shift.sum()).backward()
        self.assertIsNotNone(head.predictor[0].weight.grad)

    def test_masking_uses_predictor_change_not_target_error(self):
        head = LocalDependencyShift(4, window_size=2)
        with torch.no_grad():
            for parameter in head.predictor.parameters():
                parameter.zero_()
            head.predictor[-1].bias.fill_(1)
        effect, _ = head.intervention_effect(torch.randn(3, 2, 4))
        self.assertTrue(torch.allclose(effect, torch.zeros_like(effect), atol=1e-6))

    def test_forward_does_not_accept_or_require_labels(self):
        torch.manual_seed(42)
        head = LCDSBoundaryHead(text_dim=8, visual_dim=4, hidden_size=8, window_size=2).eval()
        output = head(torch.randn(9, 8), torch.randn(9, 4))
        self.assertEqual(output["logits"].shape, (9, 2))
        self.assertTrue(torch.equal(output["logits"][:, 0], output["base_logits"][:, 0]))


if __name__ == "__main__":
    unittest.main()
