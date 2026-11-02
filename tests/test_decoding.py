import unittest

from segment.decoding import decode_probabilities


class DecodingTest(unittest.TestCase):
    def test_final_utterance_cannot_suppress_real_boundary(self):
        self.assertEqual(decode_probabilities([0.1, 0.7, 0.99], 0.35, 6), [0, 1, 0])

    def test_gap_is_inclusive_and_ties_prefer_earliest(self):
        self.assertEqual(decode_probabilities([0.5, 0.5, 0.5, 0.0], 0.5, 2), [1, 0, 1, 0])

    def test_empty_and_single_utterance_videos(self):
        self.assertEqual(decode_probabilities([], 0.35, 6), [])
        self.assertEqual(decode_probabilities([0.99], 0.35, 6), [0])

    def test_reject_invalid_probability(self):
        with self.assertRaises(ValueError):
            decode_probabilities([float("nan")], 0.35, 6)


if __name__ == "__main__":
    unittest.main()
