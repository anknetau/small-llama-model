#pyright: strict

from utils.common import *
from core.model import Model
from tokens.bpe import BPEReader, BPEReaderResult
import tokens.token_reader_gguf
import tokens.token_reader_numeric_lines

class NumericLinesReader(BPEReader):
    def __init__(self, reader: TextIOBase):
        self.reader = reader

    def read(self) -> BPEReaderResult:
        return tokens.token_reader_numeric_lines.read(self.reader)

class GGUFReader(BPEReader):
    def __init__(self, model: Model):
        self.model = model

    def read(self) -> BPEReaderResult:
        return tokens.token_reader_gguf.read(self.model)
