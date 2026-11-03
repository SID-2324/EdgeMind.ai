"""Regression for actual NumPy embedding vectors in Stage 5B's cosine()."""
import unittest
import numpy as np
from edgemind.retrieval.hybrid import cosine


class TestNumpyCosine(unittest.TestCase):
    def test_cosine_accepts_numpy_arrays(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(cosine(a, b), 1.0)

    def test_rejects_empty_numpy_arrays(self):
        with self.assertRaises(ValueError):
            cosine(np.array([]), np.array([]))


if __name__ == '__main__':
    unittest.main()
