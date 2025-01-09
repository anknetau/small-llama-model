#!/usr/bin/env python3

from constants import Constants
import gguf
from model import Model
from bpe import BPE

# ----

from gtokenizer import GTokenizer


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

def assert_check_model(model: Model):
    # Just some sanity checking
    assert(model.n_heads == 8)
    assert(model.eps < 1e-5)

    if model.name == "Cheng98 Llama 39m":
        assert(model.block_count == 2)
        assert(model.embedding_length == 512)
    elif model.name == "Pytorch_Model.Bin":
        assert(model.block_count == 6)
        assert(model.embedding_length == 1024)
        # model.fix()
    else:
        raise NotImplementedError("Unknown model: " + model.name)

def start():
    bpe = BPE()
    bpe.read(Constants.BPE_FLCC_CS)
    assert(bpe.start.id == 2)
    assert(bpe.end.id == 3)

    model1 = Model.load(Constants.MODEL_FLCC_CS)
    assert_check_model(model1)

    model = Model.load(Constants.MODEL_LLAMA_39)
    assert_check_model(model)


    tokenizer = GTokenizer.make(model)
    vocab_size: int = model.info["llama.vocab_size"]
    assert(vocab_size == len(tokenizer.tokens))
    # print(model.detailed_description())

    return
    bpe = BPE()
    bpe.read(Constants.BPE_FLCC_CS)
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