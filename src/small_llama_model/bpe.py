#!/usr/bin/env python3

from dataclasses import dataclass, field
from tokens import Rule, Special, Specials, GToken, IdAndString, AToken, GTType
from abc import abstractmethod

#pyright: strict

class BPEReader:
    @abstractmethod
    def read(self) -> tuple[list[AToken], Specials, int]:
        raise NotImplementedError

@dataclass
class BPE:
    gtokens: list[GToken] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)
    vocab_size: int = 0

    def __post_init__(self):
        self._cache: dict[int, AToken] = dict()

    def read(self, reader: BPEReader):
        (input, specials, vocab_size) = reader.read()
        self.gtokens = [t for t in input if isinstance(t, GToken)]
        self._process_gtokens()
        self.rules = [t for t in input if isinstance(t, Rule)]
        self.specials = specials
        self.vocab_size = vocab_size
        self._fill_index()
        self.table = self._build_simple_table()

    def _process_gtokens(self):
        # BYTE tokens are converted to NORMAL
        byteMap = [f"<0x{i:02X}>" for i in range(256)]
        self.gtokens = [self._filterGToken(gt, byteMap) for gt in self.gtokens]

    def _filterGToken(self, gt: GToken, byteMap: list[str]) -> GToken:
        if gt.type != GTType.BYTE:
                return gt
        for i, str in enumerate(byteMap):
            if str == gt.str:
                return GToken(gt.id, chr(i), gt.score, GTType.NORMAL)
        raise ValueError(f"Can't parse byte {gt.str}")


    def _build_simple_table(self):
        table: list[IdAndString] = []
        for i in range(self.vocab_size):
            r = self.find(i)
            if r is None:
                raise ValueError("ID not found " + str(i))
            table.append(IdAndString(i, self.decode_token(i)))
        return table

    def _tokens_that_prefix(self, s: str):
        return [entry for entry in self.table if s.startswith(entry.string)]

    # TODO: this approach is very simple and slow - better to use a trie or similar.
    def _find_matches(self, str: str) -> list[IdAndString]:
        result: list[IdAndString] = []
        while len(str) > 0:
            m = self._tokens_that_prefix(str)
            m.sort(key=lambda x: x.string, reverse=True)
            if len(m) == 0:
                raise ValueError("ERROR: no match for \"" + str + "\"")
            else:
                result.append(m[0])
            str = str[len(m[0].string):]
        return result

    def encode(self, str: str, max: int, fill: bool=True, start: bool=True, end: bool=False) -> list[int]:
        assert(isinstance(self.specials, Specials))
        matches = self.simple_encode(str)
        if start:
            matches.insert(0, self.specials.start.id)
        if end:
            matches.append(self.specials.end.id)
        if fill:
            assert(len(matches) <= max)
            if len(matches) < max:
                matches.extend([self.specials.padding.id] * (max - len(matches)))
        return matches

    def simple_encode(self, str: str) -> list[int]:
        matches = self._find_matches(str)
        return [m.id for m in matches]

    def _fill_index(self):
        for rule in self.rules:
            self._cache[rule.c] = rule
        for special in self.specials.all():
            self._cache[special.id] = special
        for gtoken in self.gtokens:
            self._cache[gtoken.id] = gtoken

    def find(self, id: int) -> 'None | AToken':
        return self._cache.get(id)

    def decode_token(self, id: int) -> str:
        result = self.find(id)
        assert(result is not None)
        if isinstance(result, Rule):
            return self.decode_token(result.a) + self.decode_token(result.b)
        if isinstance(result, Special):
            return result.desc
        if isinstance(result, GToken): # type: ignore
            return result.str
        assert False, "result was " + str(result)

    def decode(self, list: list[int]) -> str:
        return ''.join(self.decode_token(t) for t in list)
