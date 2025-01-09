#!/usr/bin/env python3

import unittest
from small_llama_model.util import silu, softmax_all, softmax_last
import numpy as np

#pyright: strict

class TestUtils(unittest.TestCase):
    def test_silu(self):
        input = np.array([0, 0.5, 1, 2, 3, 4])
        output = silu(input)
        expected_list = [0, 0.311229665601, 0.731058578630074, 1.76159415596, 2.85772238047, 3.92805516015]
        expected = np.array(expected_list)
        np.testing.assert_array_almost_equal(output, expected)
        self.assertEqual(output[0], expected[0]) # has to be 0

    def test_softmax(self):
        input = np.array([[1, 1, 1, 1], [0, 0, 0, 0]])        
        exp = np.array([[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]])
        np.testing.assert_array_almost_equal(softmax_last(input), exp)
        k1 = np.e/(4*np.e+4)
        k2 = 1/(4*np.e+4)
        exp2 = np.array([[k1, k1, k1, k1], [k2, k2, k2, k2]])
        np.testing.assert_array_almost_equal(softmax_all(input), exp2)

if __name__ == "__main__":
    unittest.main()