from model import Model

#pyright: strict

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
