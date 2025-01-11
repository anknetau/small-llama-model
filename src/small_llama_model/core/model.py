#pyright: strict

import third.pygguf as pygguf
import math
from utils.common import *
from utils.utils import SomeNPArray, silu, softmax_all, apply_temp

@dataclass
class Tensor:
    name: str
    weight: SomeNPArray
    def __str__(self):
        return f"{self.name} {type(self.weight)} {self.weight.shape}"

@dataclass
class Block:
    attn_q: Tensor
    attn_k: Tensor
    attn_v: Tensor
    attn_output: Tensor
    ffn_gate: Tensor
    ffn_up: Tensor
    ffn_down: Tensor
    attn_norm: Tensor
    ffn_norm: Tensor

    def all_tensors(self):
        return [self.attn_q, self.attn_k, self.attn_v, self.attn_output,
                self.ffn_gate, self.ffn_up, self.ffn_down, self.attn_norm, self.ffn_norm]

    @staticmethod
    def make(dictionary: dict[str, Tensor], i: int):
        def blk(idx: int, name: str):
            return f"blk.{idx}.{name}.weight"
        return Block(
            attn_q = dictionary[blk(i, "attn_q")],
            attn_k = dictionary[blk(i, "attn_k")],
            attn_v = dictionary[blk(i, "attn_v")],
            attn_output = dictionary[blk(i, "attn_output")],
            ffn_gate = dictionary[blk(i, "ffn_gate")],
            ffn_up = dictionary[blk(i, "ffn_up")],
            ffn_down = dictionary[blk(i, "ffn_down")],
            attn_norm = dictionary[blk(i, "attn_norm")],
            ffn_norm = dictionary[blk(i, "ffn_norm")],
        )

@dataclass
class Model:
    token_embd: Tensor
    output: Tensor
    output_norm: Tensor
    blocks: list[Block]
    info: dict[str, Any]
    block_count: int
    embedding_length: int
    n_heads: int
    eps: float
    name: str
    
    @staticmethod
    def load(reader: BufferedReader) -> 'Model':
        info, tensorinfo = pygguf.load_gguf(reader)

        tensors: list[Tensor] = []

        for name in tensorinfo:
            weight = pygguf.load_gguf_tensor(reader, tensorinfo, name)
            tensor = Tensor(name, weight)
            tensors.append(tensor)

        dictionary: dict[str, Tensor] = {tensor.name: tensor for tensor in tensors}
        bc = info["llama.block_count"]

        return Model(
            token_embd = dictionary["token_embd.weight"],
            output = dictionary["output.weight"],
            output_norm = dictionary["output_norm.weight"],
            blocks = [Block.make(dictionary, i) for i in range(bc)],
            info = info,
            block_count = bc,
            embedding_length = info["llama.embedding_length"],
            n_heads = info["llama.attention.head_count"],
            eps = info["llama.attention.layer_norm_rms_epsilon"],
            name = info["general.name"]
        )

    def all_tensors(self):
        result = [self.token_embd, self.output, self.output_norm]
        for block in self.blocks:
            result.extend(block.all_tensors())
        return result

    def detailed_description(self):
        result = "Info:\n"
        for key, value in self.info.items():
            result += f"{key} {repr(value)[:100]}\n"
        
        all_tensors = self.all_tensors()

        result += "\nBlocks:\n"
        for tensor in all_tensors:
            result += tensor.name + "\n"

        return result


    def _fix(self, attn: Tensor):
        shape = attn.weight.shape
        weights = attn.weight

        # if ".attn_k." in name or ".attn_q." in name:
            # num_heads = info["llama.attention.head_count"]
        tmp_shape = (shape[-1] // self.n_heads // 2, self.n_heads, 2, shape[0])
        weights = weights.reshape(tmp_shape)
        weights = weights.transpose(0, 2, 1, 3)
        weights = weights.reshape(shape[::-1]) # type: ignore
        attn.weight = weights

    def fix(self):
        print("WARNING: TINKERING WITH ATTN K/Q")
        # not sure if needed.
        # see https://github.com/99991/pygguf/blob/main/test.py#L38C48-L38C49
        for block in self.blocks:
            self._fix(block.attn_k)
            self._fix(block.attn_q)
        return

    def run_pass(self, input: list[int]) -> int:
        logits = forward_pass(self, input, self.block_count, self.eps)
        logits_last = logits[-1]
        temperature = 0
        print("logits_last:", logits_last)
        if temperature > 0:
            probs = softmax_all(apply_temp(logits_last, temperature))
            next_token = np.random.choice(len(probs), p=probs)
        else:
            next_token = np.argmax(logits_last).item()
        return next_token

def rms_norm(hidden: SomeNPArray, weight: SomeNPArray, eps: float):
    norm_x = (hidden ** 2).mean(-1, keepdims=True) + eps
    norm_x = hidden / np.sqrt(norm_x)
    return norm_x * weight

def rms_norm_old(hidden: SomeNPArray, weight: SomeNPArray, eps: float):
    norm_x = hidden / np.sqrt((hidden**2).mean(axis=-1, keepdims=True) + eps)
    return norm_x * weight  # elementwise multiply

def apply_rope_llama(q: SomeNPArray, k: SomeNPArray, n_heads: int, base:float=10000.0):
    """
    Applies Llama v1 Rotary Position Embeddings to Q and K.
    
    Q, K:  [seq_len, hidden_size]   (flattened = n_heads * head_dim)
    n_heads: int                    number of attention heads
    base: float                     base frequency, typically 10000.0 for Llama

    Returns:
      Q_rot, K_rot (same shape as Q, K)
    """
    seq_len, hidden_size = q.shape
    head_dim = hidden_size // n_heads
    half_dim = head_dim // 2

    # Reshape Q, K to [seq_len, n_heads, head_dim]
    q = q.reshape(seq_len, n_heads, head_dim)
    k = k.reshape(seq_len, n_heads, head_dim)

    # Positions: [0..seq_len-1]
    positions = np.arange(seq_len)[:, None]  # (seq_len, 1)  # type: ignore

    # Frequencies for each dimension in [0..half_dim-1]
    #   freq[i] = 1.0 / (base ** (2*i / head_dim))
    dim_i = np.arange(half_dim)  # [0..(half_dim-1)]  # type: ignore
    freq = 1.0 / (base ** (2.0 * dim_i / head_dim))  # shape: [half_dim]  # type: ignore

    # angles => (seq_len, half_dim)
    angles = positions * freq[None, :]  # type: ignore

    cos_ = np.cos(angles)  # shape (seq_len, half_dim)  # type: ignore
    sin_ = np.sin(angles)  # shape (seq_len, half_dim)  # type: ignore

    # Now rotate Q, K:
    Q_rot = rope_rotate_llama(q, cos_, sin_)
    K_rot = rope_rotate_llama(k, cos_, sin_)

    # Reshape back to [seq_len, hidden_size]
    Q_rot = Q_rot.reshape(seq_len, n_heads * head_dim)
    K_rot = K_rot.reshape(seq_len, n_heads * head_dim)
    return Q_rot, K_rot


def rope_rotate_llama(x: SomeNPArray, cos_: SomeNPArray, sin_: SomeNPArray):
    """
    x:     [seq_len, n_heads, head_dim]
    cos_, sin_: [seq_len, half_dim] each

    Splits x into even/odd channels for each head, rotates them by cos_/sin_ 
    following Llama's approach to RoPE.
    """
    seq_len, n_heads, head_dim = x.shape
    half_dim = head_dim // 2

    # Reshape to separate the [even, odd] pairs:
    # x => [seq_len, n_heads, half_dim, 2]
    x_reshaped = x.reshape(seq_len, n_heads, half_dim, 2)
    x_even = x_reshaped[..., 0]  # shape: (seq_len, n_heads, half_dim)
    x_odd  = x_reshaped[..., 1]

    # Broadcast cos_ and sin_ to match x_even shape:
    # cos_, sin_ => (seq_len, 1, half_dim)
    cos_ = cos_[:, None, :]
    sin_ = sin_[:, None, :]

    # Rotate:
    x_even_rot = x_even * cos_ - x_odd * sin_
    x_odd_rot  = x_odd  * cos_ + x_even * sin_

    # Reassemble:
    out = np.stack([x_even_rot, x_odd_rot], axis=-1)
    out = out.reshape(seq_len, n_heads, head_dim)
    return out

def self_attn_llama(q: SomeNPArray, k: SomeNPArray, v: SomeNPArray, n_heads: int, is_causal: bool=True):
    """
    Performs Llama-style multi-head self-attention for a single sequence
    (no batch dimension in this example).

    Q, K, V: [seq_len, hidden_size]
    n_heads: int
    is_causal: bool, True => apply triangular mask (causal).

    Returns:
      context -> [seq_len, hidden_size]
    """

    seq_len, hidden_size = q.shape
    head_dim = hidden_size // n_heads

    # Reshape into [seq_len, n_heads, head_dim]
    q = q.reshape(seq_len, n_heads, head_dim)
    k = k.reshape(seq_len, n_heads, head_dim)
    v = v.reshape(seq_len, n_heads, head_dim)

    # Compute attention scores => shape [n_heads, seq_len, seq_len]
    # We do an einsum for clarity:
    scores = np.einsum('qnd,knd->nqk', q, k) / math.sqrt(head_dim)

    # Causal mask
    if is_causal:
        # mask out "future" positions: positions j > i
        # so an upper-triangular part is masked
        # shape (seq_len, seq_len), True where j>i
        causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        # broadcast to [n_heads, seq_len, seq_len]
        scores[:, causal_mask] = -1e10

    # Softmax along last axis (keys dimension, seq_len)
    scores = scores - np.max(scores, axis=-1, keepdims=True)  # stability
    exp_scores = np.exp(scores)
    attn_probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    # shape => [n_heads, seq_len, seq_len]

    # Multiply by V: [n_heads, seq_len, head_dim]
    context = np.einsum('nqk,knd->nqd', attn_probs, v)

    # Re-transpose to [seq_len, n_heads, head_dim]
    context = context.transpose(1, 0, 2)  # [seq_len, n_heads, head_dim]

    # Finally reshape to [seq_len, hidden_size]
    context = context.reshape(seq_len, hidden_size)
    return context

def forward_pass(model: Model, tokens: list[int], num_layers: int, eps: float):
    x = model.token_embd.weight[tokens, :]

    for i in range(num_layers):
        # -- Self-attn sub-block --
        blk = model.blocks[i]
        attn_in = rms_norm(x, blk.attn_norm.weight, eps)
        q = attn_in @ blk.attn_q.weight
        k = attn_in @ blk.attn_k.weight
        v = attn_in @ blk.attn_v.weight

        # Somewhere else:
        #         Q, K = apply_rope(Q, K, i)  # shape [seq_len, hidden_size], do MHA reshape, etc.
        #         context = self_attn(Q, K, V)  
        #         attn_out = context @ blk[i].attn_output.weight

        q, k = apply_rope_llama(q, k, model.n_heads)
        attn_out = self_attn_llama(q, k, v, model.n_heads)

        x = x + attn_out  # residual

        # -- FFN sub-block --
        ffn_in = rms_norm(x, blk.ffn_norm.weight, eps)
        gate  = ffn_in @ blk.ffn_gate.weight.T
        up    = ffn_in @ blk.ffn_up.weight.T

        # swiglu
        activated = silu(gate) * up
        down = activated @ blk.ffn_down.weight.T
        x = x + down  # residual

    # final norm + output
    x = rms_norm(x, model.output_norm.weight, eps)
    logits = x @ model.output.weight.T
    return logits
