#!/usr/bin/env python3

import gguf
from model import Model
from bpe import BPE

def start():
    model = Model.load("models/flcc/flcc.model")
    # Some sanity checking
    assert(model.n_heads == 8)
    assert(model.embedding_length == 1024)
    assert(model.block_count == 6)
    assert(model.eps < 1e-5)
    # model.fix()

    print(model.detailed_description())

    bpe = BPE()
    bpe.read('models/flcc/flcc.bpe')
    # Some sanity checking
    assert(bpe.start.id == 2)
    assert(bpe.end.id == 3)
    # Strangely, the json says this:
    # "bos_token_id": 1,
    # "eos_token_id": 2,


    str = "if (token.Type != TokenType.Number && token.Type != TokenType.String)"
    str = "new Tuple<int, string> (3, "

    str = """
class Foo
{
    void Main()
    {
        bool x """

    str = "bool x = "
    print("input:", str.encode())
    print("some expected:", bpe.simple_encode("true;"))
    print("some expected:", bpe.simple_encode("false;"))

    tokens = bpe.encode(str, model.embedding_length, fill=False, start=True, end=False)

    print("tokens:", tokens)

    result = model.run_pass(tokens)

    print("result:", bpe.decode_token(result))

    # for i in range(len(str)):
    #     print(f"ID #{i} is \"" + str[i] + "\"")


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