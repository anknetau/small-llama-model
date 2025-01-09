#pyright: strict

from tokens import Special, Specials, GToken, GTType, AToken
from model import Model
from typing import Any, Callable

# This file implements the GGML tokeniser within GGUF

# https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

# tokenizer.ggml.pre 'default'

PREFIX = "tokenizer.ggml."

def read(model: Model):
    named: Callable[[str], str] = lambda s: PREFIX + s
    info: Callable[[str], Any] = lambda s: model.info[named(s)]
    info_int: Callable[[str], int] = lambda s: model.info[named(s)]

    tokenizer_name = info("model")
    assert(tokenizer_name == "llama")

    tokens: list[str] = info("tokens")
    token_types: list[int] = info("token_type")
    scores: list[float] = info("scores")

    assert(len(tokens) == len(scores) == len(token_types))

    makeToken: Callable[[int], GToken] = lambda i: GToken(i, tokens[i], scores[i], GTType.make(token_types[i]))
    gtokens: list[AToken] = [makeToken(i) for i in range(len(tokens))]
    makeSpecial: Callable[[AToken], Special] = lambda t: Special(t.id, t.str) # type: ignore

    unknown = makeSpecial(gtokens[info_int("unknown_token_id")])
    start = makeSpecial(gtokens[info_int("bos_token_id")])
    end = makeSpecial(gtokens[info_int("eos_token_id")])
    # Not used:
    # tokenizer.ggml.separator_token_id: uint32: Separator token
    # tokenizer.ggml.padding_token_id: uint32: Padding token

    specials = Specials(
        unknown = unknown,
        start = start,
        end = end,
        padding = unknown, # TODO: not right!
        addStart = info("add_bos_token"),
        addEnd = info("add_eos_token")
    )

    vocab_size = len(tokens)
    return (gtokens, specials, vocab_size)
