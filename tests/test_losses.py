"""From nanoRL/: python -B -m unittest -v test_losses"""

import math
import unittest

import torch
import torch.nn.functional as F

from losses import token_logprobs


class TokenLogprobsTests(unittest.TestCase):
    def test_shift_selection_and_masked_gradients(self):
        logits = torch.tensor([
            [[1., 2., 3.], [3., 2., 1.], [99., -99., 10.]],
            [[1., 2., 3.], [3., 2., 1.], [-99., 99., 10.]],
        ], requires_grad=True)
        ids = torch.tensor([[0, 2, 1], [1, 0, 2]])
        actual = token_logprobs(logits, ids)
        log_normalizer = math.log(math.exp(1) + math.exp(2) + math.exp(3))
        expected = torch.tensor([[3., 2.], [1., 1.]]) - log_normalizer
        torch.testing.assert_close(actual, expected)

        # Full-sequence masks are shifted to the same targets as the scores.
        mask = torch.tensor([[False, True, False], [False, False, True]])[:, 1:]
        reference = -F.cross_entropy(
            logits[:, :-1].transpose(1, 2), ids[:, 1:], reduction="none"
        )
        expected_grad, = torch.autograd.grad(-reference[mask].sum(), logits)
        (-actual[mask].sum()).backward()
        torch.testing.assert_close(logits.grad, expected_grad)
        active = torch.cat([mask, torch.zeros(2, 1, dtype=torch.bool)], dim=1)
        self.assertTrue((logits.grad[~active] == 0).all())
        self.assertTrue((logits.grad[active].abs().sum(-1) > 0).all())

    def test_float32_normalization_for_low_precision_inputs(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                logits = torch.tensor(
                    [[[1000., 1005., -1000.], [0., 0., 0.]]],
                    dtype=dtype, requires_grad=True,
                )
                actual = token_logprobs(logits, torch.tensor([[0, 2]]))
                self.assertEqual(actual.shape, (1, 1))
                self.assertEqual(actual.dtype, torch.float32)
                self.assertTrue(torch.isfinite(actual).all())
                self.assertLess(actual.item(), -1000)
                actual.sum().backward()
                self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
