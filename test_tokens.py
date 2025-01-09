#!/usr/bin/env python3

import unittest
from bpe import BPE

#pyright: strict

class TestExample(unittest.TestCase):
    def setUp(self):
        self.bpe = BPE()
        self.bpe.read('models/flcc/flcc.bpe')

    def test_something(self):
        bpe = self.bpe
        self.assertEqual(bpe.start.id, 2)
        self.assertEqual(bpe.end.id, 3)
        simple_encode = bpe.simple_encode("bool x = true;")
        self.assertEqual(simple_encode, [450, 4315, 649])
        filled = bpe.encode("bool x = true;", 10, fill=True, start=True, end=True)
        self.assertEqual(filled, [2, 450, 4315, 649, 3, 0, 0, 0, 0, 0])
        back_to_string = bpe.decode([450, 4315, 649])
        self.assertEqual(back_to_string, "bool x = true;")

if __name__ == "__main__":
    unittest.main()