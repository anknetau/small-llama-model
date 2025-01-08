#!/usr/bin/env python3

from dataclasses import dataclass
import gguf
import numpy as np

# TODO: constants should come from elsewhere
EPS=1e-6
LEN=1024 # llama.embedding_length
BLOCK_COUNT=6 # llama.block_count
N_HEADS=8 # llama.attention.head_count

@dataclass
class Tensor:
    name: str
    weight: object
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

    @staticmethod
    def make(dictionary, i):
        def blk(idx, name):
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
        )

@dataclass
class Model:
    token_embd: Tensor
    output: Tensor
    output_norm: Tensor
    blocks: list[Block]
    @staticmethod
    def load(filename):
        tensors = read_tensors(filename)
        dictionary = {tensor.name: tensor for tensor in tensors}
        return Model(
            token_embd = dictionary["token_embd.weight"],
            output = dictionary["output.weight"],
            output_norm = dictionary["output_norm.weight"],
            blocks = [Block.make(dictionary, i) for i in range(BLOCK_COUNT)]
        )

def read_tensors(filename):
    tensors = []
    with open(filename, "rb") as f:
        info, tensorinfo = gguf.load_gguf(f)

        for key, value in info.items():
            print(f"{key:30} {repr(value)[:100]}")

        for name in tensorinfo:
            weight = gguf.load_gguf_tensor(f, tensorinfo, name)
            tensor = Tensor(name, weight)
            tensors.append(tensor)
            print("Loaded", tensor)
    return tensors



# def forward_pass(tokens):
#     # 1) Embeddings
#     x = token_embd.weight[tokens, :]  # shape [seq_len, hidden_size]

#     for i in range(num_layers):
#         # -- Self-attn sub-block --
#         attn_in = rms_norm(x, blk[i].attn_norm.weight)
#         Q = attn_in @ blk[i].attn_q.weight
#         K = attn_in @ blk[i].attn_k.weight
#         V = attn_in @ blk[i].attn_v.weight

#         Q, K = apply_rope(Q, K, i)  # shape [seq_len, hidden_size], do MHA reshape, etc.
#         context = self_attn(Q, K, V)  
#         attn_out = context @ blk[i].attn_output.weight
#         x = x + attn_out  # residual

#         # -- FFN sub-block --
#         ffn_in = rms_norm(x, blk[i].ffn_norm.weight)
#         gate  = ffn_in @ blk[i].ffn_gate.weight.T
#         up    = ffn_in @ blk[i].ffn_up.weight.T
#         # swiglu
#         activated = np.silu(gate) * up
#         down = activated @ blk[i].ffn_down.weight.T
#         x = x + down  # residual

#     # final norm + output
#     x = rms_norm(x, output_norm.weight)
#     logits = x @ output.weight.T  # shape [seq_len, vocab_size]
#     return logits

def rms_norm(hidden, weight, eps=EPS):
    # hidden: [seq_len, hidden_size]
    # weight: [hidden_size]
    norm_x = hidden / np.sqrt((hidden**2).mean(axis=-1, keepdims=True) + eps)
    return norm_x * weight  # elementwise multiply

import numpy as np

def apply_rope(Q, K, n_heads, base=10000.0):
    """
    Q, K: [seq_len, hidden_size]  (flattened = n_heads * head_dim)
    n_heads: number of attention heads
    base: RoPE base, often 10000.0 in Llama
    
    Returns Q_rot, K_rot (same shapes as Q/K).
    """
    seq_len, hidden_size = Q.shape
    head_dim = hidden_size // n_heads

    # Reshape Q, K to [seq_len, n_heads, head_dim]
    Q = Q.reshape(seq_len, n_heads, head_dim)
    K = K.reshape(seq_len, n_heads, head_dim)

    # Build position indices: [0..seq_len-1]
    positions = np.arange(seq_len)[:, None]  # shape (seq_len, 1)

    # For each dimension i in [0..head_dim-1], 
    # Llama uses a frequency that often looks like base^(2*i/dim). We'll do a simplified version:
    half_dim = head_dim // 2
    freq_seq = 1.0 / (base ** (2 * (np.arange(half_dim) / float(head_dim))))

    # angles: shape (seq_len, half_dim)
    angles = positions * freq_seq[None, :]

    # Compute cos/sin, shape => (seq_len, half_dim)
    cos_ = np.cos(angles)
    sin_ = np.sin(angles)

    # Rotate Q, K on their even/odd channels
    Q_rot = rope_rotate(Q, cos_, sin_)
    K_rot = rope_rotate(K, cos_, sin_)

    # Reshape back to [seq_len, hidden_size]
    Q_rot = Q_rot.reshape(seq_len, n_heads*head_dim)
    K_rot = K_rot.reshape(seq_len, n_heads*head_dim)
    return Q_rot, K_rot


def rope_rotate(x, cos_, sin_):
    """
    x: [seq_len, n_heads, head_dim]
    cos_, sin_: [seq_len, half_dim] each, broadcastable

    Applies the interleaved RoPE rotation to even/odd channels of x.
    """
    seq_len, n_heads, head_dim = x.shape
    half_dim = head_dim // 2

    # Reshape to [seq_len, n_heads, half_dim, 2]
    # so x[..., 0] and x[..., 1] map to even/odd channels
    x_reshaped = x.reshape(seq_len, n_heads, half_dim, 2)

    # Reshape cos_/sin_ to broadcast: (seq_len, 1, half_dim, 1)
    cos_ = cos_.reshape(seq_len, 1, half_dim, 1)
    sin_ = sin_.reshape(seq_len, 1, half_dim, 1)

    # x_even = x[..., 0], x_odd = x[..., 1]
    x_even = x_reshaped[..., 0]
    x_odd  = x_reshaped[..., 1]

    # RoPE formulas for each position p:
    # x_even_rot[p] = x_even[p]*cos(p) - x_odd[p]*sin(p)
    # x_odd_rot[p]  = x_odd[p]*cos(p)  + x_even[p]*sin(p)
    x_even_rot = x_even * cos_ - x_odd * sin_
    x_odd_rot  = x_odd  * cos_ + x_even * sin_

    # Combine back
    out = np.stack([x_even_rot, x_odd_rot], axis=-1)
    out = out.reshape(seq_len, n_heads, head_dim)
    return out

import numpy as np
import math

def self_attn(Q, K, V, n_heads, is_causal=True):
    """
    Q, K, V: [seq_len, hidden_size]
    n_heads: number of heads
    is_causal: whether to apply a causal (triangular) mask
    
    Returns: context => [seq_len, hidden_size]
    """
    seq_len, hidden_size = Q.shape
    head_dim = hidden_size // n_heads

    # Reshape into [seq_len, n_heads, head_dim]
    Q = Q.reshape(seq_len, n_heads, head_dim)
    K = K.reshape(seq_len, n_heads, head_dim)
    V = V.reshape(seq_len, n_heads, head_dim)

    # Compute attention scores => [n_heads, seq_len, seq_len]
    # shape: Q: [q, n, d], K: [k, n, d]
    # einsum 'qnd,knd->nqk' => for each head n, dot Q(q) with K(k)
    scores = np.einsum('qnd,knd->nqk', Q, K) / math.sqrt(head_dim)

    # Causal mask: block future positions
    if is_causal:
        # mask is upper triangular (positions > i are masked out)
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        # broadcast to [n_heads, seq_len, seq_len]
        scores[:, mask] = -1e10  # a very large negative number

    # Softmax along the last axis (keys dimension)
    # Subtract max for numerical stability
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    attn_probs = np.exp(scores)
    attn_probs /= np.sum(attn_probs, axis=-1, keepdims=True)  # shape [n_heads, seq_len, seq_len]

    # Multiply by V => [n_heads, q, head_dim]
    # 'nqk,knd->nqd'
    context = np.einsum('nqk,knd->nqd', attn_probs, V)

    # context => shape [n_heads, seq_len, head_dim]
    # We want [seq_len, n_heads, head_dim], so transpose
    context = context.transpose(1, 0, 2)  # => [seq_len, n_heads, head_dim]

    # Finally reshape back to [seq_len, hidden_size]
    context = context.reshape(seq_len, hidden_size)
    return context


def forward_pass(model, tokens, num_layers):
    # 1) Embeddings
    x = model.token_embd.weight[tokens, :]  # shape [seq_len, hidden_size]

    for i in range(num_layers):
        # -- Self-attn sub-block --
        blk = model.blocks[i]
        attn_in = rms_norm(x, blk.attn_norm.weight)
        Q = attn_in @ blk.attn_q.weight
        K = attn_in @ blk.attn_k.weight
        V = attn_in @ blk.attn_v.weight

        Q, K = apply_rope(Q, K, i)  # shape [seq_len, hidden_size], do MHA reshape, etc.
        context = self_attn(Q, K, V)  
        attn_out = context @ blk.attn_output.weight
        x = x + attn_out  # residual

        # -- FFN sub-block --
        ffn_in = rms_norm(x, blk.ffn_norm.weight)
        gate  = ffn_in @ blk.ffn_gate.weight.T
        up    = ffn_in @ blk.ffn_up.weight.T
        # swiglu
        activated = np.silu(gate) * up
        down = activated @ blk.ffn_down.weight.T
        x = x + down  # residual

    # final norm + output
    x = rms_norm(x, model.output_norm.weight)
    logits = x @ model.output.weight.T  # shape [seq_len, vocab_size]
    return logits

def start():
    model = Model.load("flcc.model")

    sample_input = [7338, 2128, 3049, 1136, 3380, 591, 5333, 1192]
    while len(sample_input) < LEN:
        sample_input.append(0)
    print(sample_input)

    forward_pass(model, sample_input, BLOCK_COUNT)


    # output_norm.weight
    # output.weight


start()