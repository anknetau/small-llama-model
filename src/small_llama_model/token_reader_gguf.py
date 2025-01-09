#pyright: strict

from tokens import Base, Rule, Special, Specials
from model import Model
from typing import Any, Callable

PREFIX = "tokenizer.ggml."

def read(model: Model):
    named: Callable[[str], str] = lambda s: PREFIX + s
    info: Callable[[str], Any] = lambda s: model.info[named(s)]

    tokens: list[str] = info("tokens")
    token_types: list[int] = info("token_type")
    scores: list[float] = info("scores")

    assert(len(tokens) == len(scores) == len(token_types))

    # makeToken: Callable[[int], GToken] = lambda i: GToken(i, tokens[i], scores[i], TType.make(token_types[i]))
    # gtokens = [makeToken(i) for i in range(len(tokens))]

    specials = Specials(
        unknown = info("unknown_token_id"),
        start = info("bos_token_id"),
        end = info("eos_token_id"),
        padding = info("unknown_token_id"), # TODO
        addStart = info("add_bos_token"),
        addEnd = info("add_eos_token")
    )
    bases: list[Base] = []
    rules: list[Rule] = []
    vocab_size = len(tokens)
    return (bases, rules, specials, vocab_size)

# def read(filename: str):
    # return (1, 2, 3, 4)
    # bases = process_bases(numbers[1:base_count+1])
    # rules = process_rules(numbers[base_count+1:base_count+1+rules_count])
    # last_line = numbers[base_count+1+rules_count]
    # specials = process_specials(last_line)
    # assert(len(numbers) == base_count+rules_count+2)
    # vocab_size = 16384
    # return (bases, rules, specials, vocab_size)
