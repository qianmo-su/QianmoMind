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