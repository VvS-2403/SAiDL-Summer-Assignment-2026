import torch
import torch.nn as nn
from typing import Optional


class TransformerBlock(nn.Module):
    """
    A single standard Transformer block utilizing Pre-Layer Normalization.
    Designed for dependency injection to easily swap Attention and FFN variants.

    ALiBi support
    -------------
    If an `alibi` module is supplied at construction time, the block computes
    the ALiBi bias matrix on every forward pass and adds it to the attention
    scores via the **kwargs pathway.  All other positional schemes (Sinusoidal,
    RoPE) leave `alibi=None` and the pathway is completely skipped.
    """

    def __init__(
        self,
        d_model: int,
        attention_module: nn.Module,
        ffn_module: nn.Module,
        dropout: float = 0.1,
        alibi: Optional[nn.Module] = None,
    ):
        """
        Args:
            d_model (int): The embedding dimension.
            attention_module (nn.Module): The instantiated attention class.
            ffn_module (nn.Module): The instantiated Feed-Forward class.
            dropout (float): Dropout probability.
            alibi (nn.Module | None): A shared ALiBiPositionalBias instance,
                                      or None for non-ALiBi runs.
        """
        super().__init__()

        # LayerNorms applied BEFORE the sub-layers (Pre-LN architecture)
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = attention_module

        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn = ffn_module

        self.dropout = nn.Dropout(dropout)

        # Store the ALiBi module (None if not used)
        self.alibi = alibi

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor, optional): Causal / padding mask.
            **kwargs: Caught and forwarded; kept for API compatibility.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        seq_len = x.size(1)

        # ── ALiBi bias ────────────────────────────────────────────────────────
        # Compute the bias matrix on-the-fly and inject it into the attention
        # call.  For all other positional schemes alibi is None and this block
        # is skipped entirely, adding zero overhead.
        if self.alibi is not None:
            alibi_bias = self.alibi(seq_len, seq_len, x.device)
            kwargs["alibi_bias"] = alibi_bias
        # ─────────────────────────────────────────────────────────────────────

        # Block 1: Multi-Head Attention Sub-layer
        # Path: Normalize → Attend → Dropout → Residual Add
        normalized_x = self.ln_1(x)
        attn_out = self.attn(normalized_x, mask=mask, **kwargs)
        x = x + self.dropout(attn_out)

        # Block 2: Feed-Forward Network Sub-layer
        # Path: Normalize → FFN → Dropout → Residual Add
        normalized_x2 = self.ln_2(x)
        ffn_out = self.ffn(normalized_x2)
        x = x + self.dropout(ffn_out)

        return x
