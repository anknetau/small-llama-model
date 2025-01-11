#!/usr/bin/env python3
#pyright: strict

from core.constants import Constants
from core.model import Model
from tokens.bpe import BPE
import tokens.token_reader_impl as token_reader_impl
from utils.checks import assert_check_model
from utils.utils import SomeNPArray

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
    with open(Constants.MODEL_FLCC_CS, "rb") as reader:
        flcc_model = Model.load(reader)
    assert_check_model(flcc_model)

    bpe = BPE()
    with open(Constants.BPE_FLCC_CS, 'r') as reader:
        bpe.read(token_reader_impl.NumericLinesReader(reader))
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
    with open(Constants.MODEL_LLAMA_39, "rb") as reader:
        llama_model = Model.load(reader)
    assert_check_model(llama_model)

    bpe = BPE()
    bpe.read(token_reader_impl.GGUFReader(llama_model))
    assert(bpe.vocab_size == 32000 == len(bpe.gtokens))

    print(llama_model.detailed_description())

    str = "Hello "
    tokens = bpe.encode(str, llama_model.embedding_length, fill=False, start=True, end=False)

    str2 = bpe.decode(tokens)
    print("tokens:", tokens, str2.encode())

    result = llama_model.run_pass(tokens)

    print("result:", bpe.decode_token(result))
    # for tensor in llama_model.all_tensors():
    #     print(dump_tensor_slice(tensor.weight, tensor.name))


def dump_tensor_slice(tensor: SomeNPArray, name: str, start: int|None=None, end:int|None=None) -> str:
    start = start or (0,) * tensor.ndim # type: ignore
    end = end or tuple(min(dim, 10) for dim in tensor.shape)  # Default to the first 10 elements of each dimension # type: ignore
    slices = tuple(slice(s, e) for s, e in zip(start, end)) # type: ignore
    sliced_tensor = tensor[slices]
    # print(f"Shape: {tensor.shape}, Sliced Shape: {sliced_tensor.shape}")
    # print(f"Values:\n{sliced_tensor}")
    return "\n>>> {" + name + "@" + str(tensor.shape) + "/" + str(sliced_tensor).replace("\n", "") + "}\n\n"


def start():
    check_llama()
    # check_flcc()

if __name__ == '__main__':
    start()