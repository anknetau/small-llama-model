#!/usr/bin/env python3

from constants import Constants
from model import Model
from bpe import BPE
from checks import assert_check_model
import token_reader_impl

#pyright: strict


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
    bpe.read(token_reader_impl.NumericLinesReader(Constants.BPE_FLCC_CS))
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
    bpe.read(token_reader_impl.GGUFReader(llama_model))

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