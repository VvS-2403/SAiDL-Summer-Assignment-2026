"""
core_ml/models/attention/gqa.py

Grouped-Query Attention (GQA).

Changes vs. original:
  • Added `n_layers` to __init__ for GPT-2 scaled out_proj init.
  • Added `alibi_bias` and `rope` kwargs to forward() for API consistency.
"""

import torch
import torch.nn as nn
import math
from einops import rearrange, repeat
from typing import Optional


class GroupedQueryAttention(nn.Module):
    """
    GQA: query heads are divided into groups; each group shares one KV head.
    Reduces KV-cache memory proportionally to num_groups.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        n_layers: int = 6,
        dropout: float = 0.1,
        is_causal: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.d_model     = d_model
        self.num_heads   = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups  = num_heads // num_kv_heads
        self.d_k         = d_model // num_heads
        self.is_causal   = is_causal

        self.q_proj = nn.Linear(d_model, num_heads    * self.d_k, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout  = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # GPT-2 scaled init
        for proj in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.normal_(proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.out_proj.weight, mean=0.0,
                         std=0.02 / math.sqrt(2 * n_layers))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        alibi_bias: Optional[torch.Tensor] = None,
        rope: Optional[nn.Module] = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (g d) -> b g s d", g=self.num_kv_heads)
        v = rearrange(v, "b s (g d) -> b g s d", g=self.num_kv_heads)

        # Apply RoPE before broadcasting KV to all query heads
        if rope is not None:
            # RoPE expects (B, H, T, d_head) — broadcast k/v to full heads first
            k_full = repeat(k, "b g s d -> b (g r) s d", r=self.num_groups)
            v_full = repeat(v, "b g s d -> b (g r) s d", r=self.num_groups)
            q, k_full = rope(q, k_full)
            k = k_full
            v = v_full
        else:
            # Broadcast KV heads to match query heads
            k = repeat(k, "b g s d -> b (g r) s d", r=self.num_groups)
            v = repeat(v, "b g s d -> b (g r) s d", r=self.num_groups)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if alibi_bias is not None:
            scores = scores + alibi_bias

        if self.is_causal and mask is None:
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=x.device)
            ).view(1, 1, seq_len, seq_len)
            scores = scores.masked_fill(causal_mask == 0, float("-inf"))
        elif mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = rearrange(context, "b h s d -> b s (h d)")
        return self.resid_dropout(self.out_proj(context))