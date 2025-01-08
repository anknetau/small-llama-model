# From https://github.com/likejazz/llama3.np/blob/main/llama3.py (MIT license)

from __future__ import annotations

import math
import sys
import time
from typing import TypeVar, Generic, Optional
from model import Model, Block
from bpe import BPE

import numpy as np


# Config
from typing import Optional
from dataclasses import dataclass

@dataclass
class ModelArgs:
    # @formatter:off
    # Model params for ./stories15M.model.npz
    dim: int                    = 288       # D
    n_layers: int               = 6
    n_heads: int                = 6         # QHN, HN, HD = 48
    n_kv_heads: Optional[int]   = None      # KVHN = 6
    vocab_size: int             = 32000     # VS
    max_seq_len: int            = 256       # M
    max_new_tokens: int         = 50
    norm_eps: float             = 1e-6
    max_batch_size: int         = 1
    # @formatter:on
# /config

DIM = 1024 # who knows!
MAX_SEQ_LEN = 1024
MAX_BATCH_SIZE = 1 # ? don't ask me, i just work here!
MAX_NEW_TOKENS = 50 # woot?

Shape = TypeVar("Shape")

class Array(np.ndarray, Generic[Shape]): ...


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def silu(x):
    return x * (1 / (1 + np.exp(-x)))

def compute_cos_sin_cache(head_dim: int, max_seq_len: int, base: int = 10000):
    inv_freq: Array = 1.0 / (base ** (np.arange(0, head_dim, 2)[: (head_dim // 2)] / head_dim))
    t: Array = np.arange(max_seq_len)
    freqs: Array = np.outer(t, inv_freq)

    return np.cos(freqs), np.sin(freqs)


def apply_rotary_emb(xq: Array, xk: Array,
                     freqs_cos: Array, freqs_sin: Array):
    # ["B, L or 1, QHN, HD"] -> ["B, L or 1, QHN,  HD//2, 2"]
    xqri = xq.reshape(xq.shape[:-1] + (-1, 2))
    xkri = xk.reshape(xk.shape[:-1] + (-1, 2))

    # Reshape `xq` and `xk` to match the complex representation.
    xq_r, xq_i = np.split(xqri, 2, axis=-1)
    xq_r = xq_r.squeeze(-1)
    xq_i = xq_i.squeeze(-1)

    xk_r, xk_i = np.split(xkri, 2, axis=-1)
    xk_r: Array = xk_r.squeeze(-1)
    xk_i: Array = xk_i.squeeze(-1)

    # Reshape `freqs_cos` and `freqs_sin` for broadcasting.
    freqs_cos: Array = np.expand_dims(freqs_cos, axis=(0, 2))
    freqs_sin: Array = np.expand_dims(freqs_sin, axis=(0, 2))

    # Apply rotation using real numbers.
    xq_out_r: Array = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i: Array = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r: Array = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i: Array = xk_r * freqs_sin + xk_i * freqs_cos

    # Flatten last two dimensions.
    xq_out: Array = np.stack([xq_out_r, xq_out_i], axis=-1)
    xk_out: Array = np.stack([xk_out_r, xk_out_i], axis=-1)
    xq_out: Array = xq_out.reshape(xq_out.shape[:-2] + (-1,))
    xk_out: Array = xk_out.reshape(xk_out.shape[:-2] + (-1,))

    return xq_out, xk_out


def repeat_kv(x: Array, n_rep: int):
    if n_rep == 1:
        return x
    z: Array = np.repeat(x, n_rep, axis=2)
    return z


class FeedForward:
    def __init__(self, up_weight: Array, gate_weight: Array, down_weight: Array):
        self.up_weight = up_weight.T
        self.gate_weight = gate_weight.T
        self.down_weight = down_weight.T

    def calc(self, x: Array):
        # FD = 2 * 4 * D / 3
        swish: Array = silu(x @ self.gate_weight)
        x_V: Array = x @ self.up_weight
        x: Array = swish * x_V
        x: Array = x @ self.down_weight
        return x


class RMSNorm:
    def __init__(self, weight: Array, eps: float):
        self.weight = weight
        self.eps = eps

    def calc(self, x: Array):
        z: Array = (x ** 2).mean(-1, keepdims=True) + self.eps
        z: Array = x / np.sqrt(z)
        return z * self.weight


class Attention:
    def __init__(self, q_weight: Array, k_weight: Array, v_weight: Array,
                 o_weight: Array, model: Model, bpe: BPE):
        
        self.model = model
        self.bpe = bpe

        self.n_kv_heads = model.n_heads # head_count_kv ? who knows!
        # assert args.n_heads % self.n_kv_heads == 0
        self.n_local_heads = model.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = DIM // model.n_heads # llama.rope.dimension_count ? is 128

        self.q_weight = q_weight.T
        self.k_weight = k_weight.T
        self.v_weight = v_weight.T
        self.o_weight = o_weight.T

        self.cache_k = np.zeros((MAX_BATCH_SIZE, MAX_SEQ_LEN, self.n_local_kv_heads, self.head_dim))
        self.cache_v = np.zeros((MAX_BATCH_SIZE, MAX_SEQ_LEN, self.n_local_kv_heads, self.head_dim))

    def calc(self, x: Array, start_pos: int, mask: Optional[Array],
                 freqs_cos: Array, freqs_sin: Array):
        B, L, _ = x.shape

        # QKV
        xq: Array = x @ self.q_weight
        xk: Array = x @ self.k_weight
        xv: Array = x @ self.v_weight

        xq: Array = xq.reshape(B, L, self.n_local_heads, self.head_dim)
        xk: Array = xk.reshape(B, L, self.n_local_kv_heads, self.head_dim)
        xv: Array = xv.reshape(B, L, self.n_local_kv_heads, self.head_dim)

        # RoPE #2
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # KV Cache
        self.cache_k[:B, start_pos: start_pos + L] = xk
        self.cache_v[:B, start_pos: start_pos + L] = xv
        ks: Array = self.cache_k[:B, : start_pos + L]
        vs: Array = self.cache_v[:B, : start_pos + L]

        # GQA
        xk: Array = repeat_kv(ks, self.n_rep)
        xv: Array = repeat_kv(vs, self.n_rep)

        # ["B, L, HN, HD"] -> ["B, HN, L, HD"]
        xq: Array = xq.transpose(0, 2, 1, 3)
        xk: Array = xk.transpose(0, 2, 1, 3)
        xv: Array = xv.transpose(0, 2, 1, 3)

        # Scaled Dot-Product Attention
        # ["B, HN, L or 1, HD"] @ ["B, HN, HD, L"] -> ["B, HN, L or 1, L"]
        attention: Array = xq @ xk.transpose(0, 1, 3, 2) / math.sqrt(self.head_dim)
        # `mask` is used only once at the beginning.
        if mask is not None:
            attention = attention + mask[None, None, :, :]
        attention = softmax(attention)
        output: Array = attention @ xv

        # ["B, HN, L or 1, HD"] -> ["B, L or 1, D"]
        output: Array = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output: Array = output @ self.o_weight

        return output


class TransformerBlock:
    def __init__(block: Block, self, model: Model, bpe):
        self.attention = Attention(
            block.attn_q.weight, # weight.get(f"model.layers.{layer_id}.self_attn.q_proj.weight"),
            block.attn_k.weight, # weight.get(f"model.layers.{layer_id}.self_attn.k_proj.weight"),
            block.attn_v.weight, # weight.get(f"model.layers.{layer_id}.self_attn.v_proj.weight"),
            block.attn_output.weight, # weight.get(f"model.layers.{layer_id}.self_attn.o_proj.weight"),
            model,
            bpe
        )
        
        self.feed_forward = FeedForward(
            block.ffn_up.weight, # weight.get(f"model.layers.{layer_id}.mlp.up_proj.weight"),
            block.ffn_gate.weight, # weight.get(f"model.layers.{layer_id}.mlp.gate_proj.weight"),
            block.ffn_down.weight # weight.get(f"model.layers.{layer_id}.mlp.down_proj.weight"),
        )
        self.input_layernorm = RMSNorm(
            block.ffn_norm.weight, # weight.get(f"model.layers.{layer_id}.input_layernorm.weight"),
            eps=model.eps
        )
        self.post_attention_layernorm = RMSNorm(
            block.attn_norm.weight, # weight.get(f"model.layers.{layer_id}.post_attention_layernorm.weight"),
            eps=model.eps
        )

    def calc(self, x: Array, start_pos: int, mask: Array,
                 freqs_cos: Array, freqs_sin: Array):
        # RMSNorm
        norm_x: Array = self.input_layernorm.calc(x)
        # Masked Multi-Head Attention
        h1: Array = self.attention.calc(norm_x, start_pos, mask, freqs_cos, freqs_sin)
        z = x + h1

        # RMSNorm
        norm_z = self.post_attention_layernorm.calc(z)
        # Feed Forward + SwiGLU
        h2: Array = self.feed_forward.calc(norm_z)
        out = z + h2

        return out


class Llama:
    def __init__(self, model: Model, bpe: BPE):
        self.model = model
        self.bpe = bpe

        self.tok_embedding: Array = model.token_embd.weight

        # RoPE #1
        self.freqs_cos, self.freqs_sin = compute_cos_sin_cache(DIM // model.n_heads, MAX_SEQ_LEN)

        self.layers = []
        for id in range(model.block_count):
            self.layers.append(TransformerBlock(model.blocks[id], model, bpe))

        self.norm = RMSNorm(model.output_norm.weight, eps=model.eps) # weight.get("model.norm.weight"), eps=model.eps)
        self.lm_head_weight: Array = model.output.weight.T # weight.get("lm_head.weight").T


    def calc(self, input_ids: Array, start_pos: int):
        _, L = input_ids.shape
        h: Array = self.tok_embedding[input_ids]
        # ["M, HD//2"] -> ["L or 1, HD//2"]
        freqs_cos: Array = self.freqs_cos[start_pos: start_pos + L]
        freqs_sin: Array = self.freqs_sin[start_pos: start_pos + L]

        # `mask` is generated only once at the beginning.
        mask: Array = None
        if L > 1:
            mask = np.full((L, L), float("-inf"))
            mask = np.triu(mask, k=1)
            mask = np.concatenate([np.zeros((L, start_pos)), mask], axis=1)

        # Transformer Layers
        for i, layer in enumerate(self.layers):
            h: Array = layer.calc(h, start_pos, mask, freqs_cos, freqs_sin)

        # RMSNorm
        h: Array = self.norm.calc(h)
        # Only forward the output from the last position.
        # ["B, 1, VS"] = ["B, 1(L), D"] @ ["D, VS"]
        logit: Array = h[:, [-1], :] @ self.lm_head_weight
        return logit

    def generate(self, input_ids: Array, max_new_tokens: int):
        input_ids = input_ids
        
        _, L = input_ids.shape
        # for i, curr_pos in enumerate(range(L, max_new_tokens)):
        #     print(input_ids)
        #     if i == 0:  # Prefill Phase
        #         inputs = input_ids
        #         pos = 0
        #     else:  # Decode Phase
        #         inputs = next_id
        #         pos = curr_pos
        logits: Array = self.calc(input_ids, 0)
        next_id = logits[:, -1, :].argmax(-1, keepdims=True)
        return next_id

    def generate_old(self, input_ids: Array, max_new_tokens: int):
        _, L = input_ids.shape
        for i, curr_pos in enumerate(range(L, max_new_tokens)):
            if i == 0:  # Prefill Phase
                inputs = input_ids
                pos = 0
            else:  # Decode Phase
                inputs = next_id
                pos = curr_pos
            logits: Array = self.calc(inputs, pos)
            next_id = logits[:, -1, :].argmax(-1, keepdims=True)
            yield next_id


if __name__ == '__main__':
    model = Model.load("models/flcc/flcc.model")
    # model.fix()
    bpe = BPE()
    bpe.read('models/flcc/flcc.bpe')

    llama = Llama(model, bpe)

    prompt = """
class Foo
{
    void Main()
    {
        bool x """

    prompt = "bool x = "    

    start = time.time()
    tokens = bpe.encode(prompt, model.embedding_length, fill=False, end=False)
    print(prompt)

    input_ids = np.array([tokens])
    _, L = input_ids.shape

    # result = llama.generate(input_ids, MAX_NEW_TOKENS)
    # print(result)
    # print(bpe.decode_token(result[0][0]))

    for id in llama.generate_old(input_ids, MAX_NEW_TOKENS):
        L += 1
        output_id = id[0].tolist()
        if output_id[-1] in [bpe.end.id, bpe.start.id]:
            break
        print(bpe.decode_token(output_id[-1]), end="")
        sys.stdout.flush()

    elapsed = time.time() - start
    print(f"\n\nToken count: {L}, elapsed: {elapsed:.2f}s, {round(L / elapsed)} tokens/s")