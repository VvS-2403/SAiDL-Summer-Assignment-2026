"""
core_ml/models/attention/relu_attention.py

ReLU Attention — replaces Softmax with ReLU for true sparsity.

Changes vs. original:
  • Added `n_layers` to __init__ for GPT-2 scaled out_proj init.
  • Added `alibi_bias` and `rope` kwargs to forward().
"""

import torch
import torch.nn as nn
import math
from einops import rearrange
from typing import Optional


class ReLUAttention(nn.Module):
    """
    Softmax-free attention: uses ReLU(scores) / seq_len for normalisation.
    Produces true zeros (sparse attention patterns) for negative dot-products.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        n_layers: int = 6,
        dropout: float = 0.1,
        is_causal: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self.is_causal = is_causal

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

        if self.is_causal and mask is None:
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=x.device)
            ).view(1, 1, seq_len, seq_len)
            scores = scores.masked_fill(causal_mask == 0, float("-inf"))
        elif mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # ReLU replaces Softmax; normalise by seq_len to control scale
        attn_weights = torch.relu(scores) / seq_len
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = rearrange(context, "b h s d -> b s (h d)")
        return self.resid_dropout(self.out_proj(context))