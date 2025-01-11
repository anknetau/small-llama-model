#!/usr/bin/env python3

#pyright: strict

import unittest
from small_llama_model import BPE, Constants
from small_llama_model.tokens import token_reader_impl

class TestTokens(unittest.TestCase):
    def setUp(self):
        self.bpe = BPE()
        with open(Constants.BPE_FLCC_CS, 'r') as reader:
            self.bpe.read(token_reader_impl.NumericLinesReader(reader))

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