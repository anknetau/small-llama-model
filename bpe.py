#!/usr/bin/env python3

from dataclasses import dataclass
from dataclasses import field

@dataclass
class Special:
    id: int
    desc: str
    def __str__(self):
        return f"{self.id}=\"{self.desc}\""

@dataclass
class Base:
    id: int
    char: int
    def __str__(self):
        return f"{self.id}='{self.char}'"

@dataclass
class IdAndString:
    id: int
    string: str
    def __str__(self):
        return f"{self.id}=\"{self.string}\""

@dataclass
class Rule:
    a: int
    b: int
    c: int
    def __str__(self):
        return f"{self.c}={self.a}+{self.b}"

@dataclass
class BPE:
    bases: list[Base] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)
    specials: list[Special] = field(default_factory=list)
    _baseCache: dict[int, Base] = field(default_factory=dict)
    _ruleCache: dict[int, Rule] = field(default_factory=dict)
    _specialCache: dict[int, Special] = field(default_factory=dict)
    def read(self, filename):
        numbers = self._read_lines(filename)
        base_count = numbers[0][0]
        rules_count = numbers[0][1]
        self.bases = self._process_bases(numbers[1:base_count+1])
        self.rules = self._process_rules(numbers[base_count+1:base_count+1+rules_count])
        last_line = numbers[base_count+1+rules_count]
        self._process_specials(last_line)
        assert(len(numbers) == base_count+rules_count+2)
        self._index()
        self.table = self._build_simple_table()

    def _build_simple_table(self):
        table = []
        for i in range(16384):
            r = self.find(i)
            if r is None:
                print("ERROR: ID not found " + str(i))
                return
            # print(f"ID #{i} is \"" + bpe.decode(i) + "\"")
            table.append(IdAndString(i, self.decode_token(i)))
        return table

    def _tokens_that_prefix(self, str):
        return [entry for entry in self.table if str.startswith(entry.string)]

    # TODO: this approach is very simple and slow - better to use a trie or similar.
    def _find_matches(self, str):
        result = []
        while len(str) > 0:
            m = self._tokens_that_prefix(str)
            m.sort(key=lambda x: x.string, reverse=True)
            if len(m) == 0:
                print("ERROR: no match for \"" + str + "\"")
                return
            else:
                result.append(m[0])
            str = str[len(m[0].string):]
        return result

    def encode(self, str, max, fill=True, end=False) -> list[int]:
        matches = self.simple_encode(str)
        matches.insert(0, self.start.id)
        if end:
            matches.append(self.end.id)
        if fill:
            assert(len(matches) <= max)
            if len(matches) < max:
                matches.extend([self.padding.id] * (max - len(matches)))
        return matches

    def simple_encode(self, str) -> list[int]:
        matches = self._find_matches(str)
        return [m.id for m in matches]

    def _index(self):
        for b in self.bases:
            self._baseCache[b.id] = b
        for r in self.rules:
            self._ruleCache[r.c] = r
        for s in self.specials:
            self._specialCache[s.id] = s

    def _process_bases(self, pairs):
        result = []
        for pair in pairs:
            assert len(pair) == 2
            id = pair[1]
            char = pair[0]
            result.append(Base(id, char))
        return result

    def _process_rules(self, triplets):
        result = []
        for t in triplets:
            assert len(t) == 3
            result.append(Rule(t[0], t[1], t[2]))
        return result
    
    def _process_specials(self, last_line):
        self.unknown = Special(last_line[0], "<|UNKNOWN|>")
        self.padding = Special(last_line[1], "<|PADDING|>")
        self.start = Special(last_line[2], "<|START|>")
        self.end = Special(last_line[3], "<|END|>")
        self.specials = [self.unknown, self.padding, self.start, self.end]

    def find_base(self, id):
        return self._baseCache.get(id)
    def find_rule(self, id):
        return self._ruleCache.get(id)
    def find_special(self, id):
        return self._specialCache.get(id)
    def find(self, id):
        result = self.find_base(id)
        if result is not None:
            return result
        result = self.find_special(id)
        if result is not None:
            return result
        return self.find_rule(id)

    def decode_token(self, id: int):
        result = self.find(id)
        assert(result is not None)
        if isinstance(result, Base):
            return chr(result.char) + ""
        if isinstance(result, Rule):
            return self.decode_token(result.a) + self.decode_token(result.b)
        if isinstance(result, Special):
            return result.desc
        assert False, "result was " + str(result)

    def decode(self, list):
        return ''.join(self.decode_token(t) for t in list)

    def _read_lines(self, filename):
        all_numbers = []
        with open(filename, 'r') as file:
            for line in file:
                number_strings = line.strip().split()
                numbers = [int(num) for num in number_strings]
                all_numbers.append(numbers)
        return all_numbers
