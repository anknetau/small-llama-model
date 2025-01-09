#!/usr/bin/env python3

import unittest
from bpe import BPE
from constants import Constants
import a

#pyright: strict

class TestTokens(unittest.TestCase):
    def setUp(self):
        self.bpe = BPE()
        self.bpe.read(a.NumericLinesReader(Constants.BPE_FLCC_CS))

    def test_bpe(self):
        bpe = self.bpe
        self.assertEqual(bpe.specials.start.id, 2)
        self.assertEqual(bpe.specials.end.id, 3)
        simple_encode = bpe.simple_encode("bool x = true;")
        self.assertEqual(simple_encode, [450, 4315, 649])
        filled = bpe.encode("bool x = true;", 10, fill=True, start=True, end=True)
        self.assertEqual(filled, [2, 450, 4315, 649, 3, 0, 0, 0, 0, 0])
        back_to_string = bpe.decode([450, 4315, 649])
        self.assertEqual(back_to_string, "bool x = true;")

if __name__ == "__main__":
    unittest.main()