# From https://github.com/likejazz/llama3.np/blob/main/llama3.py (MIT license)
# Some copyright (c) 2025-, modifications by Andres Kievsky

from __future__ import annotations

import math
from core.model import Model, Block
from tokens.bpe import BPE

from numpy import ndarray

from utils.common import *
from utils.checks import assert_check_model
from utils.utils import silu, softmax_last, print_single

def compute_cos_sin_cache(head_dim: int, max_seq_len: int, base: int = 10000):
    inv_freq: ndarray = 1.0 / (base ** (np.arange(0, head_dim, 2)[: (head_dim // 2)] / head_dim))
    t: ndarray = np.arange(max_seq_len)
    freqs: ndarray = np.outer(t, inv_freq)
    return np.cos(freqs), np.sin(freqs)

def apply_rotary_emb(xq: ndarray, xk: ndarray, freqs_cos: ndarray, freqs_sin: ndarray):
    # ["B, L or 1, QHN, HD"] -> ["B, L or 1, QHN,  HD//2, 2"]
    xqri = xq.reshape(xq.shape[:-1] + (-1, 2))
    xkri = xk.reshape(xk.shape[:-1] + (-1, 2))

    # Reshape `xq` and `xk` to match the complex representation.
    xq_r, xq_i = np.split(xqri, 2, axis=-1)
    xq_r = xq_r.squeeze(-1)
    xq_i = xq_i.squeeze(-1)

    xk_r, xk_i = np.split(xkri, 2, axis=-1)
    xk_r: ndarray = xk_r.squeeze(-1)
    xk_i: ndarray = xk_i.squeeze(-1)

    # Reshape `freqs_cos` and `freqs_sin` for broadcasting.
    freqs_cos: ndarray = np.expand_dims(freqs_cos, axis=(0, 2))
    freqs_sin: ndarray = np.expand_dims(freqs_sin, axis=(0, 2))

    # Apply rotation using real numbers.
    xq_out_r: ndarray = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i: ndarray = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r: ndarray = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i: ndarray = xk_r * freqs_sin + xk_i * freqs_cos

    # Flatten last two dimensions.
    xq_out: ndarray = np.stack([xq_out_r, xq_out_i], axis=-1)
    xk_out: ndarray = np.stack([xk_out_r, xk_out_i], axis=-1)
    xq_out: ndarray = xq_out.reshape(xq_out.shape[:-2] + (-1,))
    xk_out: ndarray = xk_out.reshape(xk_out.shape[:-2] + (-1,))

    return xq_out, xk_out


def repeat_kv(x: ndarray, n_rep: int):
    if n_rep == 1:
        return x
    z: ndarray = np.repeat(x, n_rep, axis=2)
    return z

def feed_forward(up_weight: ndarray, gate_weight: ndarray, down_weight: ndarray, in_x: ndarray):
    # FD = 2 * 4 * D / 3
    swish: ndarray = silu(in_x @ gate_weight)
    x_V: ndarray = in_x @ up_weight
    x: ndarray = swish * x_V
    x1: ndarray = x @ down_weight
    return x1

def rms_norm(weight: ndarray, eps: float, x: ndarray):
    z: ndarray = (x ** 2).mean(-1, keepdims=True) + eps
    z: ndarray = x / np.sqrt(z)
    return z * weight

def calc_attention(block: Block, x: ndarray, start_pos: int, mask: Optional[ndarray],
                   freqs_cos: ndarray, freqs_sin: ndarray, model: Model):
    n_kv_heads = model.n_heads # head_count_kv ? who knows!
    # assert args.n_heads % self.n_kv_heads == 0
    n_local_heads = model.n_heads
    n_local_kv_heads = n_kv_heads
    n_rep = n_local_heads // n_local_kv_heads
    
    DIM = model.embedding_length
    head_dim = DIM // model.n_heads # llama.rope.dimension_count ? is 128

    q_weight: ndarray = block.attn_q.weight.T
    k_weight: ndarray = block.attn_k.weight.T
    v_weight: ndarray = block.attn_v.weight.T
    o_weight: ndarray = block.attn_output.weight.T

    # 1, 1024, 8, 128
    MAX_BATCH_SIZE = 1
    MAX_SEQ_LEN = model.embedding_length
    cache_k = np.zeros((MAX_BATCH_SIZE, MAX_SEQ_LEN, n_local_kv_heads, head_dim))
    cache_v = np.zeros((MAX_BATCH_SIZE, MAX_SEQ_LEN, n_local_kv_heads, head_dim))

    B, L, _ = x.shape

    # QKV
    xq: ndarray = x @ q_weight
    xk: ndarray = x @ k_weight
    xv: ndarray = x @ v_weight

    xq: ndarray = xq.reshape(B, L, n_local_heads, head_dim)
    xk: ndarray = xk.reshape(B, L, n_local_kv_heads, head_dim)
    xv: ndarray = xv.reshape(B, L, n_local_kv_heads, head_dim)

    # RoPE #2
    xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

    # KV Cache
    cache_k[:B, start_pos: start_pos + L] = xk
    cache_v[:B, start_pos: start_pos + L] = xv
    ks: ndarray = cache_k[:B, : start_pos + L]
    vs: ndarray = cache_v[:B, : start_pos + L]

    # GQA
    xk: ndarray = repeat_kv(ks, n_rep)
    xv: ndarray = repeat_kv(vs, n_rep)

    # ["B, L, HN, HD"] -> ["B, HN, L, HD"]
    xq: ndarray = xq.transpose(0, 2, 1, 3)
    xk: ndarray = xk.transpose(0, 2, 1, 3)
    xv: ndarray = xv.transpose(0, 2, 1, 3)

    # Scaled Dot-Product Attention
    # ["B, HN, L or 1, HD"] @ ["B, HN, HD, L"] -> ["B, HN, L or 1, L"]
    attention: ndarray = xq @ xk.transpose(0, 1, 3, 2) / math.sqrt(head_dim)
    # `mask` is used only once at the beginning.
    if mask is not None:
        attention = attention + mask[None, None, :, :]
    attention = softmax_last(attention)
    output: ndarray = attention @ xv

    # ["B, HN, L or 1, HD"] -> ["B, L or 1, D"]
    output: ndarray = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    output: ndarray = output @ o_weight

    return output

def tranformer_calc(block: Block, x: ndarray, start_pos: int, mask: ndarray,
                freqs_cos: ndarray, freqs_sin: ndarray, eps: float, model):
    # RMSNorm - input_layernorm(x) -- ank
    norm_x: ndarray = rms_norm(block.ffn_norm.weight, # weight.get(f"model.layers.{layer_id}.input_layernorm.weight"),
                                eps, x)

    # Masked Multi-Head Attention
    h1: ndarray = calc_attention(block, norm_x, start_pos, mask, freqs_cos, freqs_sin, model)
    z = x + h1

    # RMSNorm - post_attention_layernorm(z) -- ank
    norm_z = rms_norm(block.attn_norm.weight, # weight.get(f"model.layers.{layer_id}.post_attention_layernorm.weight"),
                      eps, z)

    # Feed Forward + SwiGLU
    h2: ndarray = feed_forward(block.ffn_up.weight.T, # weight.get(f"model.layers.{layer_id}.mlp.up_proj.weight"),
                               block.ffn_gate.weight.T, # weight.get(f"model.layers.{layer_id}.mlp.gate_proj.weight"),
                               block.ffn_down.weight.T, # weight.get(f"model.layers.{layer_id}.mlp.down_proj.weight"),
                               norm_z)
    out = z + h2

    return out


class LlamaThirdParty:
    def __init__(self, model: Model, bpe: BPE):
        self.model = model
        self.bpe = bpe

        self.tok_embedding: ndarray = model.token_embd.weight

        # RoPE #1
        DIM = model.embedding_length
        MAX_SEQ_LEN = model.embedding_length
        self.freqs_cos, self.freqs_sin = compute_cos_sin_cache(DIM // model.n_heads, MAX_SEQ_LEN)

        # self.norm = RMSNorm(model.output_norm.weight, eps=model.eps) # weight.get("model.norm.weight"), eps=model.eps)
        self.lm_head_weight: ndarray = model.output.weight.T # weight.get("lm_head.weight").T


    def calc(self, input_ids: ndarray, start_pos: int):
        # print("start_pos", start_pos)
        _, L = input_ids.shape
        h: ndarray = self.tok_embedding[input_ids]
        # ["M, HD//2"] -> ["L or 1, HD//2"]
        freqs_cos: ndarray = self.freqs_cos[start_pos: start_pos + L]
        freqs_sin: ndarray = self.freqs_sin[start_pos: start_pos + L]

        # `mask` is generated only once at the beginning.
        # here, L goes from 3 in the first call to 1 afterwards -- ank
        mask: ndarray = None
        if L > 1:
            mask = np.full((L, L), float("-inf"))
            mask = np.triu(mask, k=1)
            mask = np.concatenate([np.zeros((L, start_pos)), mask], axis=1)

        # Transformer Layers
        for id in range(self.model.block_count):
            block = self.model.blocks[id]
            h2: ndarray = tranformer_calc(block, h, start_pos, mask, freqs_cos, freqs_sin, self.model.eps, self.model)
            h = h2

        # RMSNorm - norm(h) -- ank
        h3: ndarray = rms_norm(self.model.output_norm.weight, self.model.eps, h) # weight.get("model.norm.weight"), eps=model.eps)
        h = h3
        # Only forward the output from the last position.
        # ["B, 1, VS"] = ["B, 1(L), D"] @ ["D, VS"]
        logit: ndarray = h[:, [-1], :] @ self.lm_head_weight
        return logit

    def generate(self, input_ids: ndarray, max_new_tokens: int):
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
        logits: ndarray = self.calc(input_ids, 0)
        next_id = logits[:, -1, :].argmax(-1, keepdims=True)
        return next_id

    def run_pass(self, tokens: list[int], bpe: BPE):
        input_ids = np.array([tokens])
        _, l = input_ids.shape

        # how many tokens to stop at
        MAX_NEW_TOKENS = 50
        for id in self.generate_old(input_ids, MAX_NEW_TOKENS):
            print(id)
            l += 1
            output_id = id[0].tolist()
            if output_id[-1] in [bpe.specials.end.id, bpe.specials.start.id]:
                break
            print_single(bpe.decode_token(output_id[-1]))

    def generate_old(self, input_ids: ndarray[Any, Any], max_new_tokens: int) -> Generator[ndarray[Any, Any], None, None]:
        _, L = input_ids.shape
        for i, curr_pos in enumerate(range(L, max_new_tokens)):
            if i == 0:  # Prefill Phase
                inputs = input_ids
                pos = 0
            else:  # Decode Phase
                inputs = next_id
                pos = curr_pos
            logits: ndarray = self.calc(inputs, pos)
            next_id = logits[:, -1, :].argmax(-1, keepdims=True)
            yield next_id
