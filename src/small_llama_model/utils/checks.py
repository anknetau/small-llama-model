#pyright: strict

from core.model import Model
from tokens.bpe import BPE
from core.constants import Constants

def assert_check_model(model: Model):
    # Just some sanity checking
    assert(model.n_heads == 8)
    assert(model.eps < 1e-5)

    if model.name == "Cheng98 Llama 39m":
        assert(model.block_count == 2)
        assert(model.embedding_length == 512)
    elif model.name == "Pytorch_Model.Bin":
        assert(model.block_count == 6)
        assert(model.embedding_length == 1024)
        # model.fix()
    else:
        raise NotImplementedError("Unknown model: " + model.name)

def assert_check_bpe(bpe: BPE):
    if bpe.name == Constants.TYPE_BPE_NUMERIC:
        # Some sanity checking
        assert(bpe.specials.start.id == 2)
        assert(bpe.specials.end.id == 3)
        # Strangely, the json says this:
        # "bos_token_id": 1,
        # "eos_token_id": 2,
        assert(bpe.vocab_size == 16384)
    elif bpe.name == Constants.TYPE_BPE_GGML:
        assert(bpe.vocab_size == 32000 == len(bpe.gtokens))
    else:
        assert(False), "Unknown BPE type " + bpe.name