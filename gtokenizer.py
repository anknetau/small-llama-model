from dataclasses import dataclass
from enum import Enum
from model import Model
from typing import Any
from typing import Callable

#pyright: strict

class GType(Enum):
    NORMAL=1
    UNKNOWN=2
    CONTROL=3
    USER_DEFINED=4
    UNUSED=5
    BYTE=6

    @classmethod
    def is_valid(cls, value: int) -> bool:
        """Check if a value is a valid GType member."""
        return value in cls._value2member_map_

    @classmethod
    def make(cls, value: int) -> 'GType':
        if cls.is_valid(value):
            return GType(value)
        else:
            raise ValueError(f"Invalid token type {value}")

@dataclass
class GToken:
    c: str
    id: int
    score: float
    type: GType

    # - `tokenizer.ggml.token_type: array[int32]`: The token type (1=normal, 2=unknown, 3=control, 4=user defined, 5=unused, 6=byte). If present, it must have the same length and index as `tokens`.

PREFIX = "tokenizer.ggml."

@dataclass
class Specials:
    unknown: int
    bos: int
    eos: int
    add_bos: bool
    add_eos: bool

@dataclass
class GTokenizer:
    model: Model
    tokens: list[GToken]
    specials: Specials

    @staticmethod
    def make(model: Model):
        named: Callable[[str], str] = lambda s: PREFIX + s
        info: Callable[[str], Any] = lambda s: model.info[named(s)]

        tokens: list[str] = info("tokens")
        token_types: list[int] = info("token_type")
        scores: list[float] = info("scores")

        assert(len(tokens) == len(scores) == len(token_types))

        makeToken: Callable[[int], GToken] = lambda i: GToken(tokens[i], i, scores[i], GType.make(token_types[i]))
        gtokens = [makeToken(i) for i in range(len(tokens))]

        specials = Specials(
            unknown = info("unknown_token_id"),
            bos = info("bos_token_id"),
            eos = info("eos_token_id"),
            add_bos = info("add_bos_token"),
            add_eos = info("add_eos_token")
        )

        return GTokenizer(model, gtokens, specials)
