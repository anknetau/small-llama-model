from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

#pyright: strict

AToken: TypeAlias = "Base | Rule | GToken | Special"

# A GGUF token
@dataclass
class GToken:
    id: int
    str: str
    score: float
    type: 'GTType'

# A character with an ID associated with it
@dataclass
class Base:
    id: int
    char: int
    def __str__(self):
        return f"{self.id}='{self.char}'"

# A special token
@dataclass
class Special:
    id: int
    desc: str
    def __str__(self):
        return f"{self.id}=\"{self.desc}\""

# A merge rule: "concatenate token a with b to get token c"
@dataclass
class Rule:
    a: int
    b: int
    c: int
    def __str__(self):
        return f"{self.c}={self.a}+{self.b}"

# A Token ID and some string associated with it
@dataclass
class IdAndString:
    id: int
    string: str
    def __str__(self):
        return f"{self.id}=\"{self.string}\""

@dataclass
class Specials:
    unknown: Special
    start: Special
    end: Special
    padding: Special
    addStart: bool = False
    addEnd: bool = False

    def all(self):
        return [self.unknown, self.start, self.end, self.padding]        

class GTType(Enum):
    NORMAL=1
    UNKNOWN=2
    CONTROL=3
    USER_DEFINED=4
    UNUSED=5
    BYTE=6

    @classmethod
    def is_valid(cls, value: int) -> bool:
        return value in cls._value2member_map_

    @classmethod
    def make(cls, value: int):
        if not cls.is_valid(value):
            raise ValueError(f"Invalid token type {value}")
        return GTType(value)

