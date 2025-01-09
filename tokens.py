from dataclasses import dataclass

#pyright: strict

# A special token
@dataclass
class Special:
    id: int
    desc: str
    def __str__(self):
        return f"{self.id}=\"{self.desc}\""

@dataclass
class Specials:
    unknown: Special
    start: Special
    end: Special
    padding: Special
    addStart: bool | None
    addEnd: bool | None

    def all(self):
        return [self.unknown, self.start, self.end, self.padding]        


# A character with an ID associated with it
@dataclass
class Base:
    id: int
    char: int
    def __str__(self):
        return f"{self.id}='{self.char}'"

# A Token ID and some string associated with it
@dataclass
class IdAndString:
    id: int
    string: str
    def __str__(self):
        return f"{self.id}=\"{self.string}\""

# A merge rule: "concatenate token a with b to get token c"
@dataclass
class Rule:
    a: int
    b: int
    c: int
    def __str__(self):
        return f"{self.c}={self.a}+{self.b}"
