#pyright: strict

from os.path import dirname

class Constants:
    MODEL_BASE_DIR = dirname(__file__) + "/../../models/"
    BPE_FLCC_CS = MODEL_BASE_DIR + "flcc-cs/flcc.bpe"
    MODEL_FLCC_CS = MODEL_BASE_DIR + "flcc-cs/flcc.model"
    MODEL_LLAMA_39 = MODEL_BASE_DIR + "llama-39m-Q5_K_M.gguf"