from bpe import BPEReader
from model import Model

#pyright: strict

import token_reader_numeric_lines
import token_reader_gguf
from tokens import Specials, AToken

class NumericLinesReader(BPEReader):
    def __init__(self, filename: str):
        self.filename = filename

    def read(self) -> tuple[list[AToken], Specials, int]:
        return token_reader_numeric_lines.read(self.filename)

class GGUFReader(BPEReader):
    def __init__(self, model: Model):
        self.model = model

    def read(self) -> tuple[list[AToken], Specials, int]:
        return token_reader_gguf.read(self.model)
