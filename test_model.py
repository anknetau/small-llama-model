#!/usr/bin/env python3

import gguf
from model import Model
from bpe import BPE

def start():
    model = Model.load("flcc.model")

    bpe = BPE()
    bpe.read('flcc.bpe')

    str = "if (token.Type != TokenType.Number && token.Type != TokenType.String)"
    str = "VariantParser.ErrorAndMsg? err = null;"
    tokens = bpe.encode(str, model.LEN)

    # tokens = [7338, 2128, 3049, 1136, 3380, 591, 5333, 1192]
    # while len(tokens) < model.LEN:
    #     tokens.append(0)
    print(tokens)

    # result = model.do_it(tokens)
    # print(result)


    
    # while len(ids) < 1024:
    #     ids.append(bpe.padding.id)

    # ids = [m.id for m in matches]

    # while len(ids) < 1024:
    #     ids.append(bpe.padding.id)
    # print(ids)

    

    # for i in range(len(str)):
    #     print(f"ID #{i} is \"" + str[i] + "\"")


    return 
    for r in bpe.rules:
        rr = bpe.find(r.a)
        if rr is None:
            print(f"not found: {r.a}")
            return

        rr = bpe.find(r.b)
        if rr is None:
            print(f"not found: {r.b}")
            return

        print(f"ID #{r.c} is \"" + bpe.decode(r.c) + "\"")

start()