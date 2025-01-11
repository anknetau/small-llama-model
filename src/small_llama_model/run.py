#!/usr/bin/env python3
#pyright: strict

from core.constants import Constants
from core.model import Model
from tokens.bpe import BPE
import tokens.token_reader_impl as token_reader_impl
from utils.checks import assert_check_model
from utils.common import *
from third.llama3 import LlamaThirdParty
from utils.prompts import prompt_cs, prompt_text

def start():
    check_llama()
    check_flcc()
    check_llama_third()

# checks

def check_llama_third():
    model, bpe, prompt = load_llama()
    llama_third = LlamaThirdParty(model, bpe)
    tokens = bpe.encode(prompt, model.embedding_length, fill=False, start=True, end=False)
    llama_third.run_pass(tokens, bpe)

def check_flcc():
    flcc_model, bpe, prompt = load_flcc()
    print("input:", str.encode(prompt))
    tokens = bpe.encode(prompt, flcc_model.embedding_length, fill=False, start=True, end=False)
    print("tokens:", tokens)
    result = flcc_model.run_pass(tokens)
    print("result:", bpe.decode_token(result))

def check_llama():
    llama_model, bpe, prompt = load_llama()
    tokens = bpe.encode(prompt, llama_model.embedding_length, fill=False, start=True, end=False)
    str2 = bpe.decode(tokens)
    print("tokens:", tokens, str2.encode())
    result = llama_model.run_pass(tokens)
    print("result:", bpe.decode_token(result))

# Loaders

def load_llama():
    print("Llama")
    with open(Constants.MODEL_LLAMA_39, "rb") as reader:
        llama_model = Model.load(reader)
    assert_check_model(llama_model)
    bpe = BPE()
    bpe.read(token_reader_impl.GGUFReader(llama_model))
    assert(bpe.vocab_size == 32000 == len(bpe.gtokens))
    return llama_model, bpe, prompt_text()

def load_flcc():
    print("FLCC")
    with open(Constants.MODEL_FLCC_CS, "rb") as reader:
        flcc_model = Model.load(reader)
    assert_check_model(flcc_model)
    # print(flcc_model.detailed_description())

    bpe = BPE()
    with open(Constants.BPE_FLCC_CS, 'r') as reader:
        bpe.read(token_reader_impl.NumericLinesReader(reader))

    # Some sanity checking
    assert(bpe.specials.start.id == 2)
    assert(bpe.specials.end.id == 3)
    # Strangely, the json says this:
    # "bos_token_id": 1,
    # "eos_token_id": 2,

    return flcc_model, bpe, prompt_cs()


if __name__ == '__main__':
    start()


    # result = llama.generate(input_ids, MAX_NEW_TOKENS)
    # print(result)
    # print(bpe.decode_token(result[0][0]))
    # print("another token:", bpe.simple_encode("true;"))
    # print("another token:", bpe.simple_encode("false;"))


    # print(llama_model.detailed_description())
    #
    # tokens = bpe.encode(str, llama_model.embedding_length, fill=False, start=True, end=False)
    #
    # str2 = bpe.decode(tokens)
    # print("tokens:", tokens, str2.encode())
    #
    # result = llama_model.run_pass(tokens)
    #
    # print("result:", bpe.decode_token(result))
