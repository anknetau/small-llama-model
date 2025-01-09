#pyright: strict

from tokens import Special, Specials, GToken, GTType, AToken
from model import Model
from typing import Any, Callable

# tokenizer.ggml.model 'llama'
# tokenizer.ggml.pre 'default'
# tokenizer.ggml.tokens ['<unk>', '<s>', '</s>', '<0x00>', '<0x01>', '<0x02>', '<0x03>', '<0x04>', '<0x05>', '<0x06>', '<0x0
# tokenizer.ggml.scores [-1000.0, -1000.0, -1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.
# tokenizer.ggml.token_type [3, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 

PREFIX = "tokenizer.ggml."

def read(model: Model):
    named: Callable[[str], str] = lambda s: PREFIX + s
    info: Callable[[str], Any] = lambda s: model.info[named(s)]
    info_int: Callable[[str], int] = lambda s: model.info[named(s)]

    tokens: list[str] = info("tokens")
    token_types: list[int] = info("token_type")
    scores: list[float] = info("scores")

    assert(len(tokens) == len(scores) == len(token_types))

    makeToken: Callable[[int], GToken] = lambda i: GToken(i, tokens[i], scores[i], GTType.make(token_types[i]))
    gtokens: list[AToken] = [makeToken(i) for i in range(len(tokens))]
    makeSpecial: Callable[[GToken], Special] = lambda gtoken: Special(gtoken.id, gtoken.str)

    unknown = makeSpecial(gtokens[info_int("unknown_token_id")])
    start = makeSpecial(gtokens[info_int("bos_token_id")])
    end = makeSpecial(gtokens[info_int("eos_token_id")])

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
