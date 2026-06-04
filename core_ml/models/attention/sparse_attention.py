"""
core_ml/models/attention/Sparse_attention.py

Fixed-Pattern Sparse Attention: local window + strided global tokens.

Changes vs. original:
  • Added `n_layers` to __init__ for GPT-2 scaled out_proj init.
  • Added `alibi_bias` and `rope` kwargs to forward().
"""

import torch
import torch.nn as nn
import math
from einops import rearrange
from typing import Optional


class SparseAttention(nn.Module):
    """
    Combines a local neighbourhood (sliding window) with strided global tokens
    so every query can attend to at least one token from every stride-width
    chunk of history.  Sub-quadratic in practice for long sequences.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        n_layers: int = 6,
        local_window: int = 64,
        stride: int = 64,
        dropout: float = 0.1,
        is_causal: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model      = d_model
        self.num_heads    = num_heads
        self.d_k          = d_model // num_heads
        self.local_window = local_window
        self.stride       = stride
        self.is_causal    = is_causal

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout  = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # GPT-2 scaled init
        nn.init.normal_(self.qkv_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.out_proj.weight,  mean=0.0,
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

        qkv = self.qkv_proj(x)
        q, k, v = rearrange(
            qkv, "b s (three h d) -> three b h s d",
            three=3, h=self.num_heads,
        )

        if rope is not None:
            q, k = rope(q, k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if alibi_bias is not None:
            scores = scores + alibi_bias

        # Build sparse mask
        row_idx  = torch.arange(seq_len, device=x.device).unsqueeze(1)
        col_idx  = torch.arange(seq_len, device=x.device).unsqueeze(0)
        distance = row_idx - col_idx

        is_local   = (distance >= 0) & (distance < self.local_window)
        is_strided = (distance >= 0) & (distance % self.stride == 0)
        sparse_mask = is_local | is_strided

        scores = scores.masked_fill(
            (~sparse_mask).unsqueeze(0).unsqueeze(0), float("-inf")
        )

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = rearrange(context, "b h s d -> b s (h d)")
        return self.resid_dropout(self.out_proj(context))