from transformers import PretrainedConfig


class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )
        
# RMSNorm
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
    
    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)
    
# RoPE + YaRN
import math
from typing import Optional

def build_rope_freqs(dim: int, rope_base: float = 10000.0):
    # arange:[0, 2, 4, ... , dim]
    # freqs.shape = [dim/2] --- 1-D vector
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))
    return freqs

def apply_yarn_scaling(
    freqs: torch.Tensor,
    dim: int,
    rope_base: float,
    orig_max: int,
    factor: float,
    beta_fast: float,
    beta_slow: float
):
    def inv_dim(beta):
        return (
            dim * math.log(orig_max / (beta * 2 * math.pi))
            / (2 * math.log(rope_base))
        )
        
    low = max(math.floor(inv_dim(beta_fast)), 0)
    high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
    
    idx = torch.arange(dim // 2, device=freqs.device).float()
    
    ramp = torch.clamp(
        (idx - low) / max(high - low, 0.001),
        0,
        1
    )
    
    freqs = freqs * (1 - ramp + ramp / factor)
    
    return freqs

def build_rope_cache(dim: int, end: int, rope_base: float = 10000.0, rope_scaling: Optional[dict] = None):
    freqs = build_rope_freqs(dim, rope_base)
    
    attn_factor = 1.0
    
    if rope_scaling is not None:
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 16)
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        attn_factor = rope_scaling.get("attention_factor", 1.0)
        
        if end > orig_max:
            freqs = apply_yarn_scaling(
                freqs=freqs,
                dim=dim,
                rope_base=rope_base,
                orig_max=orig_max,
                factor=factor,
                beta_fast=beta_fast,
                beta_slow=beta_slow,
            )
    
    # end -> the count of tokens
    # t.shape = [end] --- 1-D vector
    t = torch.arange(end, device=freqs.device)
    # angles.shape = [dim/2, end] --- 2-D matrix
    # angles[i, j] means the rotate angle on pos i and group j
    angles = torch.outer(t, freqs).float()
    
    # e.g. we have a vector [x0, x1, x2, x3, x4, x5]
    # half-split:(x0, x3)--θ0,(x1,x4)--θ1,(x2,x5)--θ2
    # [x0, x1, x2, x3, x4, x5] -- [c0, c1, c2, c0, c1, c2]
    # that is why we need to use 'cat'
    cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1)
    sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1)
    
    return cos, sin

def rotate_half(x):
    return torch.cat(
        # [x0, x1, x2, x3, x4, x5] -> [-x3, -x4, -x5, x0, x1, x2]
        (-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]),
        dim=-1
    )
    
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    if position_ids is None:
        seq_len = q.shape[-2]
        cos = cos[:seq_len]
        sin = sin[:seq_len]
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    else:
        cos = cos[position_ids][:, None, :, :]
        sin = sin[position_ids][:, None, :, :]
    
    # rotate
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.dim = config.hidden_size // config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.rope_scaling = config.rope_scaling
        
        cos, sin = build_rope_cache(
            dim=self.dim,
            end=self.max_position_embeddings,
            rope_base=self.rope_theta,
            rope_scaling=self.rope_scaling
        )
        
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
    
    def forward(self, q, k, position_ids=None):
        return apply_rotary_pos_emb(
            q,
            k,
            self.cos_cached,
            self.sin_cached,
            position_ids=position_ids
        )

# GQA
def repeat_kv(x: torch.Tensor, n_rep: int):
    bsz, num_kv_heads, seq_len, head_dim = x.shape
    
    if n_rep == 1:
        return x
    
    x = x[:, :, None, :, :]
    x = x.expand(bsz, num_kv_heads, n_rep, seq_len, head_dim)
    
    return x.reshape(bsz, num_kv_heads * n_rep, seq_len, head_dim)

def make_causal_mask(seq_len, device, dtype):
    mask = torch.full(
        (seq_len, seq_len),
        torch.finfo(dtype).min,
        device=device
    )
    mask = torch.triu(mask, diagonal=1)
    return mask[None, None, :, :]

# hidden_states: [B, T, H]

# q_proj -> [B, T, num_heads * head_dim]
# k_proj -> [B, T, num_kv_heads * head_dim]
# v_proj -> [B, T, num_kv_heads * head_dim]

# reshape + transpose:
# q -> [B, num_heads, T, head_dim]
# k -> [B, num_kv_heads, T, head_dim]
# v -> [B, num_kv_heads, T, head_dim]

# RoPE(q, k)

# repeat_kv:
# k/v -> [B, num_heads, T, head_dim]

# attention:
# q @ k^T -> [B, num_heads, T, T]
# attn @ v -> [B, num_heads, T, head_dim]

# merge heads:
# [B, T, H]

# o_proj
class MokioMindAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.dropout = config.dropout
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(config)
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None
    ):
        bsz, seq_len, _ = hidden_states.shape
        
        q_states = self.q_proj(hidden_states)
        k_states = self.k_proj(hidden_states)
        v_states = self.v_proj(hidden_states)
        
        q_states = q_states.view(
            bsz, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        k_states = k_states.view(
            bsz, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        
        v_states = v_states.view(
            bsz, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        
        k_states = repeat_kv(k_states, self.num_kv_groups)
        v_states = repeat_kv(v_states, self.num_kv_groups)
        
        attn_weight = torch.matmul(
            q_states,
            k_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weight = attn_weight + attention_mask
        
        attn_weight = torch.softmax(
            attn_weight,
            dim=-1,
            dtype=torch.float32
        ).to(q_states.dtype)
        
        attn_weight = nn.functional.dropout(
            attn_weight,
            p=self.dropout,
            training=self.training
        )
        
        attn_output = torch.matmul(attn_weight, v_states)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(
            bsz,
            seq_len,
            self.num_heads * self.head_dim
        )
        
        attn_output = self.o_proj(attn_output)
        
        return attn_output

# FFN
class MokioMindMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        
        if config.intermediate_size is None:
            self.intermediate_size = 4 * config.hidden_size
        else:
            self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=False
        )
        
        self.up_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=False
        )
        
        self.down_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=False
        )
        
        self.act_fn = nn.SiLU()
    
    def forward(self, x):
        return self.down_proj(
            self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        )

# Residual Connection
class MokioMindDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.self_attn = MokioMindAttention(config)
        self.mlp = MokioMindMLP(config)
        
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps
        )
        
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps
        )
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None
    ):
        residual = hidden_states
        
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        hidden_states = self.mlp(hidden_states)
        
        hidden_states = residual + hidden_states
        
        return hidden_states

# Model
class MokioMindModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size
        )
        
        self.layers = nn.ModuleList(
            [
                MokioMindDecoderLayer(config)
                for _ in range(config.num_hidden_layers)
            ]
        )
        
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps
        )
    
    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None
    ):
        bsz, seq_len = input_ids.shape
        
        if position_ids is None:
            position_ids = torch.arange(
                seq_len,
                device=input_ids.device
            ).unsqueeze(0).expand(bsz, -1)
        
        if attention_mask is None:
            attention_mask = make_causal_mask(
                seq_len=seq_len,
                device=input_ids.device,
                dtype=torch.float32
            )
        
        hidden_states = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
        
        hidden_states = self.norm(hidden_states)
        
        return hidden_states