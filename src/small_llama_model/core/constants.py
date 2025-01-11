#pyright: strict

from os.path import dirname

class Constants:
    MODEL_BASE_DIR = dirname(__file__) + "/../../../models/"
    BPE_FLCC_CS = MODEL_BASE_DIR + "flcc-cs/flcc.bpe"
    MODEL_FLCC_CS = MODEL_BASE_DIR + "flcc-cs/flcc.model"
    MODEL_LLAMA_39 = MODEL_BASE_DIR + "llama-39m-Q8_0.gguf"
    URL_LLAMA_39 = "https://huggingface.co/tensorblock/llama-39m-GGUF/blob/main/llama-39m-Q8_0.gguf"
