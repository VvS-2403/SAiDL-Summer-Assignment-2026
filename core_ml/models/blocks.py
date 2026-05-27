"""
core_ml/models/blocks.py

Single standard Transformer block (Pre-LN) with dependency injection for:
  • ALiBi / Relative positional bias  (via `alibi` constructor arg)
  • RoPE                               (via `rope`  constructor arg)

Both are forwarded into the attention module through its forward() kwargs,
keeping the Transformer shell and block API completely clean.
"""

import torch
import torch.nn as nn
from typing import Optional


class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer block.

    Positional scheme injection
    ---------------------------
    alibi (nn.Module | None):
        An ALiBiPositionalBias or RelativePositionalBias instance.
        The block calls alibi(seq_len, seq_len, device) on every forward pass
        and injects the result as `alibi_bias` into the attention kwargs.

    rope (nn.Module | None):
        A RotaryPositionalEmbedding instance.
        Passed directly as `rope=...` into the attention forward() so the
        attention module can rotate Q and K after head-splitting (the only
        mathematically correct place to apply RoPE).

    For vanilla Sinusoidal runs, both are None and there is zero overhead.
    """

    def __init__(
        self,
        d_model: int,
        attention_module: nn.Module,
        ffn_module: nn.Module,
        dropout: float = 0.1,
        alibi: Optional[nn.Module] = None,
        rope:  Optional[nn.Module] = None,
    ):
        """
        Args:
            d_model           : Embedding dimension.
            attention_module  : Instantiated attention class.
            ffn_module        : Instantiated Feed-Forward class.
            dropout           : Dropout probability.
            alibi             : ALiBi or RelativeBias module, or None.
            rope              : RotaryPositionalEmbedding module, or None.
        """
        super().__init__()

        self.ln_1   = nn.LayerNorm(d_model)
        self.attn   = attention_module

        self.ln_2   = nn.LayerNorm(d_model)
        self.ffn    = ffn_module

        self.dropout = nn.Dropout(dropout)

        self.alibi = alibi   # None for Sinusoidal / RoPE runs
        self.rope  = rope    # None for Sinusoidal / ALiBi / Relative runs

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x    : (batch_size, seq_len, d_model)
            mask : optional causal / padding mask
        Returns:
            (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)

        # ── Positional bias (ALiBi or learned Relative Bias) ──────────────
        # Computed on-the-fly and injected into attention via alibi_bias kwarg.
        # Skipped entirely for Sinusoidal and RoPE runs (alibi is None).
        if self.alibi is not None:
            kwargs["alibi_bias"] = self.alibi(seq_len, seq_len, x.device)

        # ── RoPE ──────────────────────────────────────────────────────────
        # Pass the module itself into attention so it can rotate Q and K
        # after head-splitting (the module is stateless at inference time).
        if self.rope is not None:
            kwargs["rope"] = self.rope

        # ── Attention sub-layer (Pre-LN) ──────────────────────────────────
        x = x + self.dropout(self.attn(self.ln_1(x), mask=mask, **kwargs))

        # ── FFN sub-layer (Pre-LN) ────────────────────────────────────────
        x = x + self.dropout(self.ffn(self.ln_2(x)))

        return x