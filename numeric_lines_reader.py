from tokens import Base, Rule, Special, Specials

#pyright: strict

def read_numeric_lines(filename: str):
    all_numbers: list[list[int]] = []
    with open(filename, 'r') as file:
        for line in file:
            number_strings = line.strip().split()
            numbers: list[int] = [int(num) for num in number_strings]
            all_numbers.append(numbers)
    return all_numbers

def process_bases(pairs: list[list[int]]):
    result: list[Base] = []
    for pair in pairs:
        assert len(pair) == 2
        id = pair[1]
        char = pair[0]
        result.append(Base(id, char))
    return result

def process_rules(triplets: list[list[int]]):
    result: list[Rule] = []
    for t in triplets:
        assert len(t) == 3
        result.append(Rule(t[0], t[1], t[2]))
    return result

def process_specials(last_line: list[int]):
    return Specials(
        unknown = Special(last_line[0], "<|UNKNOWN|>"),
        padding = Special(last_line[1], "<|PADDING|>"),
        start = Special(last_line[2], "<|START|>"),
        end = Special(last_line[3], "<|END|>"),
        addStart=None,
        addEnd=None
    )

