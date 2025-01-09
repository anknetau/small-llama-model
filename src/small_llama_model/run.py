#!/usr/bin/env python3

from constants import Constants
from model import Model
from bpe import BPE, BPEReader
from checks import assert_check_model

#pyright: strict

# tokenizer.ggml.model 'llama'
# tokenizer.ggml.pre 'default'
# tokenizer.ggml.tokens ['<unk>', '<s>', '</s>', '<0x00>', '<0x01>', '<0x02>', '<0x03>', '<0x04>', '<0x05>', '<0x06>', '<0x0
# tokenizer.ggml.scores [-1000.0, -1000.0, -1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
# tokenizer.ggml.token_type [3, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 
# tokenizer.ggml.bos_token_id 1
# tokenizer.ggml.eos_token_id 2
# tokenizer.ggml.unknown_token_id 0
# tokenizer.ggml.add_bos_token True
# tokenizer.ggml.add_eos_token False

# READERS - TODO: moveme
import numeric_lines_reader
import token_reader_gguf
from tokens import Base, Rule, Specials
from typing import Any, TypeAlias, Callable

class NumericLinesReader(BPEReader):
    def __init__(self, filename: str):
        self.filename = filename

    def read(self) -> tuple[list[Base], list[Rule], Specials, int]:
        return numeric_lines_reader.read(self.filename)

# /READERS



def sample_input():
    str = "if (token.Type != TokenType.Number && token.Type != TokenType.String)"
    str = "new Tuple<int, string> (3, "

    str = """
class Foo
{
    void Main()
    {
        bool x """

    str = "bool x = "
    return str

def check_flcc():
    print("FLCC")
    flcc_model = Model.load(Constants.MODEL_FLCC_CS)
    assert_check_model(flcc_model)

    bpe = BPE()
    bpe.read(NumericLinesReader(Constants.BPE_FLCC_CS))
    # Some sanity checking
    assert(bpe.specials.start.id == 2)
    assert(bpe.specials.end.id == 3)
    # Strangely, the json says this:
    # "bos_token_id": 1,
    # "eos_token_id": 2,

    str = sample_input()

    print("input:", str.encode())
    print("another token:", bpe.simple_encode("true;"))
    print("another token:", bpe.simple_encode("false;"))

    tokens = bpe.encode(str, flcc_model.embedding_length, fill=False, start=True, end=False)

    print("tokens:", tokens)

    result = flcc_model.run_pass(tokens)

    print("result:", bpe.decode_token(result))


def check_llama():
    print("Llama")
    llama_model = Model.load(Constants.MODEL_LLAMA_39)
    assert_check_model(llama_model)

    bpe = BPE()
    # bpe.read(llama_model)

    # tokenizer = GTokenizer.make(llama_model)
    # vocab_size: int = llama_model.info["llama.vocab_size"]
    # assert(vocab_size == len(tokenizer.tokens))
    # print(model.detailed_description())

    # tokens = bpe.encode(str, llama_model.embedding_length, fill=False, start=True, end=False)

    # print("tokens:", tokens)

    # result = llama_model.run_pass(tokens)

    # print("result:", bpe.decode_token(result))


def start():
    check_flcc()
    check_llama()

    return 
    for r in bpe.rules:
        rr = bpe.find(r.a)
        if rr is None:
            print(f"not found: {r.a}")
            return

        rr = bpe.find(r.b)
        if rr is None:
            print(f"not found: {r.b}")
            return

        print(f"ID #{r.c} is \"" + bpe.decode(r.c) + "\"")

start()