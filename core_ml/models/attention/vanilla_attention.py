"""
core_ml/models/attention/vanilla_attention.py

Standard Multi-Head Self-Attention.

Key fixes vs. the original:
  1. The class was accidentally exported as `Mul` — now only `MultiHeadAttention`.
  2. RoPE support: if a `rope` module is passed via kwargs, it rotates Q and K
     INSIDE the attention forward pass (the only correct place to apply RoPE).
  3. Relative bias: if `alibi_bias` is passed (used for both ALiBi and the
     learned RelativePositionalBias), it is added to scores before softmax.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Self-Attention with GPT-2 scaled initialisation,
    ALiBi / relative-bias support, and optional RoPE rotation.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.1,
        is_causal: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be exactly divisible by n_heads."

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.is_causal = is_causal

        # Combined Q, K, V projection
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        # Output projection back to residual stream
        self.out_proj  = nn.Linear(d_model, d_model)

        self.attn_dropout  = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # ── GPT-2 scaled weight initialisation ────────────────────────────
        nn.init.normal_(self.qkv_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.qkv_proj.bias)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        alibi_bias: torch.Tensor = None,   # ALiBi OR RelativeBias tensor
        rope: nn.Module = None,            # RotaryPositionalEmbedding instance
        **kwargs,                           # absorbs any extra kwargs cleanly
    ) -> torch.Tensor:
        """
        Args:
            x          : (B, T, C)
            mask       : optional causal / padding bool mask (B,1,T,T) or (1,1,T,T)
            alibi_bias : (1, H, T, T) pre-computed positional bias (ALiBi or Relative)
            rope       : a RotaryPositionalEmbedding module; rotates Q and K in-place
        """
        B, T, C = x.size()

        # 1. Project → Q, K, V
        qkv = self.qkv_proj(x)                                # (B, T, 3C)
        q, k, v = qkv.split(self.d_model, dim=2)

        # Reshape to (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # 2. Apply RoPE rotation to Q and K (if provided)
        #    RoPE MUST be applied here, after head-splitting, not to the flat token embedding.
        if rope is not None:
            q, k = rope(q, k)

        # 3. Scaled dot-product scores
        scale  = 1.0 / math.sqrt(self.d_head)
        scores = (q @ k.transpose(-2, -1)) * scale            # (B, H, T, T)

        # 4. Add positional bias (ALiBi or learned Relative Bias)
        if alibi_bias is not None:
            scores = scores + alibi_bias

        # 5. Causal masking
        if self.is_causal:
            if mask is None:
                mask = torch.tril(
                    torch.ones(T, T, dtype=torch.bool, device=x.device)
                ).view(1, 1, T, T)
            scores = scores.masked_fill(~mask, float("-inf"))
        elif mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # 6. Softmax + dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 7. Weighted sum of values
        out = attn_weights @ v                                 # (B, H, T, d_head)

        # 8. Re-assemble heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out))


# ── Backward-compat alias (the original file exported "Mul" by mistake) ──────
Mul = MultiHeadAttention