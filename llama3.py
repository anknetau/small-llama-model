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
# /config

# Info:
# general.size_label '111M'
# llama.vocab_size 16384
# llama.context_length 1536
# llama.embedding_length 1024
# llama.block_count 6
# llama.feed_forward_length 2816
# llama.rope.dimension_count 128
# llama.attention.head_count 8
# llama.attention.head_count_kv 8
# llama.attention.layer_norm_rms_epsilon 9.999999974752427e-07
# llama.rope.freq_base 10000.0
# general.quantization_version 2



DIM = 1024 # who knows!
MAX_SEQ_LEN = 1024
MAX_BATCH_SIZE = 1 # ? don't ask me, i just work here!

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

def feed_forward(up_weight: Array, gate_weight: Array, down_weight: Array, in_x: Array):
        # FD = 2 * 4 * D / 3
        swish: Array = silu(in_x @ gate_weight)
        x_V: Array = in_x @ up_weight
        x: Array = swish * x_V
        x1: Array = x @ down_weight
        return x1

def rms_norm(weight: Array, eps: float, x: Array):
        z: Array = (x ** 2).mean(-1, keepdims=True) + eps
        z: Array = x / np.sqrt(z)
        return z * weight

def calc_attention(block: Block, x: Array, start_pos: int, mask: Optional[Array], freqs_cos: Array, freqs_sin: Array):
        n_kv_heads = model.n_heads # head_count_kv ? who knows!
        # assert args.n_heads % self.n_kv_heads == 0
        n_local_heads = model.n_heads
        n_local_kv_heads = n_kv_heads
        n_rep = n_local_heads // n_local_kv_heads
        head_dim = DIM // model.n_heads # llama.rope.dimension_count ? is 128

        q_weight: Array = block.attn_q.weight.T
        k_weight: Array = block.attn_k.weight.T
        v_weight: Array = block.attn_v.weight.T
        o_weight: Array = block.attn_output.weight.T

        # 1, 1024, 8, 128
        cache_k = np.zeros((MAX_BATCH_SIZE, MAX_SEQ_LEN, n_local_kv_heads, head_dim))
        cache_v = np.zeros((MAX_BATCH_SIZE, MAX_SEQ_LEN, n_local_kv_heads, head_dim))

        B, L, _ = x.shape

        # QKV
        xq: Array = x @ q_weight
        xk: Array = x @ k_weight
        xv: Array = x @ v_weight

        xq: Array = xq.reshape(B, L, n_local_heads, head_dim)
        xk: Array = xk.reshape(B, L, n_local_kv_heads, head_dim)
        xv: Array = xv.reshape(B, L, n_local_kv_heads, head_dim)

        # RoPE #2
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # KV Cache
        cache_k[:B, start_pos: start_pos + L] = xk
        cache_v[:B, start_pos: start_pos + L] = xv
        ks: Array = cache_k[:B, : start_pos + L]
        vs: Array = cache_v[:B, : start_pos + L]

        # GQA
        xk: Array = repeat_kv(ks, n_rep)
        xv: Array = repeat_kv(vs, n_rep)

        # ["B, L, HN, HD"] -> ["B, HN, L, HD"]
        xq: Array = xq.transpose(0, 2, 1, 3)
        xk: Array = xk.transpose(0, 2, 1, 3)
        xv: Array = xv.transpose(0, 2, 1, 3)

        # Scaled Dot-Product Attention
        # ["B, HN, L or 1, HD"] @ ["B, HN, HD, L"] -> ["B, HN, L or 1, L"]
        attention: Array = xq @ xk.transpose(0, 1, 3, 2) / math.sqrt(head_dim)
        # `mask` is used only once at the beginning.
        if mask is not None:
            attention = attention + mask[None, None, :, :]
        attention = softmax(attention)
        output: Array = attention @ xv

        # ["B, HN, L or 1, HD"] -> ["B, L or 1, D"]
        output: Array = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output: Array = output @ o_weight

        return output


# class TransformerBlock:
#     def __init__(self, block: Block, model: Model, bpe):
#         self.block = block;  

def tranformer_calc(block: Block, x: Array, start_pos: int, mask: Array,
                freqs_cos: Array, freqs_sin: Array, eps: float):
    # RMSNorm
    # norm_x: Array = self.input_layernorm.calc(x)
    norm_x: Array = rms_norm(block.ffn_norm.weight, eps, x)
    # self.input_layernorm = rms_norm(
    #     block.ffn_norm.weight, # weight.get(f"model.layers.{layer_id}.input_layernorm.weight"),
    #     eps=model.eps
    # )

    # Masked Multi-Head Attention
    h1: Array = calc_attention(block, norm_x, start_pos, mask, freqs_cos, freqs_sin)
    z = x + h1

    # RMSNorm
    # norm_z = self.post_attention_layernorm.calc(z)
    norm_z = rms_norm(block.attn_norm.weight, eps, z)
    # self.post_attention_layernorm = rms_norm(
    #     block.attn_norm.weight, # weight.get(f"model.layers.{layer_id}.post_attention_layernorm.weight"),
    #     eps=model.eps
    # )

    # Feed Forward + SwiGLU
    # h2: Array = self.feed_forward.calc(norm_z)
    h2: Array = feed_forward(block.ffn_up.weight.T, block.ffn_gate.weight.T, block.ffn_down.weight.T, norm_z)
    #     block.ffn_up.weight, # weight.get(f"model.layers.{layer_id}.mlp.up_proj.weight"),
    #     block.ffn_gate.weight, # weight.get(f"model.layers.{layer_id}.mlp.gate_proj.weight"),
    #     block.ffn_down.weight # weight.get(f"model.layers.{layer_id}.mlp.down_proj.weight"),

    out = z + h2

    return out


class Llama:
    def __init__(self, model: Model, bpe: BPE):
        self.model = model
        self.bpe = bpe

        self.tok_embedding: Array = model.token_embd.weight

        # RoPE #1
        self.freqs_cos, self.freqs_sin = compute_cos_sin_cache(DIM // model.n_heads, MAX_SEQ_LEN)

        # self.norm = RMSNorm(model.output_norm.weight, eps=model.eps) # weight.get("model.norm.weight"), eps=model.eps)
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
        for id in range(model.block_count):
            block = model.blocks[id]
            h2: Array = tranformer_calc(block, h, start_pos, mask, freqs_cos, freqs_sin, model.eps)
            h = h2

        # RMSNorm
        # h: Array = self.norm.calc(h)
        h3: Array = rms_norm(model.output_norm.weight, model.eps, h) # weight.get("model.norm.weight"), eps=model.eps)
        h = h3
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
    print(model.detailed_description())

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

    MAX_NEW_TOKENS = 100
    for id in llama.generate_old(input_ids, MAX_NEW_TOKENS):
        L += 1
        output_id = id[0].tolist()
        if output_id[-1] in [bpe.end.id, bpe.start.id]:
            break
        print(bpe.decode_token(output_id[-1]), end="")
        sys.stdout.flush()

    elapsed = time.time() - start
    print(f"\n\nToken count: {L}, elapsed: {elapsed:.2f}s, {round(L / elapsed)} tokens/s")