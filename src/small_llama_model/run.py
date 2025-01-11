#!/usr/bin/env python3
#pyright: strict

from core.constants import Constants
from core.model import Model
from tokens.bpe import BPE
import tokens.token_reader_impl as token_reader_impl
from utils.checks import assert_check_model, assert_check_bpe
from utils.common import *
from third.llama3 import LlamaThirdParty
from utils.prompts import prompt_cs, prompt_text

def start():
    check(*load_llama())
    check(*load_flcc())
    check(*load_llama(), True)


# checks

def check(model: Model, bpe: BPE, prompt: str, useThirdParty: bool=False):
    print("input:", str.encode(prompt))
    tokens = bpe.encode(prompt, model.embedding_length, fill=False, start=True, end=False)
    print("tokens:", tokens)
    if useThirdParty:
        llama_third = LlamaThirdParty(model, bpe)
        llama_third.run_pass(tokens, bpe)
    else:
        result = model.run_pass(tokens)
        print("result:", bpe.decode_token(result))

# Loaders

def load_llama():
    print("* Loaded Llama")
    with open(Constants.MODEL_LLAMA_39, "rb") as reader:
        llama_model = Model.load(reader)
    assert_check_model(llama_model)
    bpe = BPE()
    bpe.read(token_reader_impl.GGUFReader(llama_model))
    assert_check_bpe(bpe)
    return llama_model, bpe, prompt_text()

def load_flcc():
    print("* Loaded FLCC")
    with open(Constants.MODEL_FLCC_CS, "rb") as reader:
        flcc_model = Model.load(reader)
    assert_check_model(flcc_model)
    bpe = BPE()
    with open(Constants.BPE_FLCC_CS, 'r') as reader:
        bpe.read(token_reader_impl.NumericLinesReader(reader))
    assert_check_bpe(bpe)
    return flcc_model, bpe, prompt_cs()


if __name__ == '__main__':
    start()


    # print(flcc_model.detailed_description())

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
