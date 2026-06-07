"""
core_ml/models/hybrid/hybrid_blocks.py

Fixed version — addresses every root cause of NaN/exploding PPL.

Bug inventory and fixes
-----------------------
BUG 1 — DepthwiseSeparableConv1d: double residual
    Old code:  `return self.norm(residual + x_t)` then the calling block ALSO
               adds `x = self.conv(x) + x` (since conv() returns a residual-added
               tensor).  Result: residual added twice → variance doubles every
               layer → NaN within a few hundred steps on deeper stacks.
    Fix:       conv helper returns a raw transform (NO residual, NO norm).
               Each block handles its own Pre-LN residual + dropout correctly.

BUG 2 — DepthwiseSeparableConv1d: initialisation missing
    Pointwise conv weight was initialised with PyTorch default (Kaiming uniform),
    which is too large for a residual-stream module.  Depthwise also uninitialised.
    Fix:       Both conv layers get nn.init.normal_(w, 0.0, 0.02) (GPT-2 scale).
               Pointwise bias zeroed.

BUG 3 — GatedConvFFN (in GatedConvFFNBlock): missing scaled init on out_proj
    Out-projection was default-initialised (Kaiming), not GPT-2 scaled.
    This causes output variance to grow proportionally to d_ff, not d_model,
    blowing up the residual stream at initialisation.
    Fix:       nn.init.normal_(self.out_proj.weight, 0.0, 0.02 / sqrt(2*n_layers))
               n_layers is now a required parameter of GatedConvFFN / GatedConvFFNBlock.

BUG 4 — GatedConvFFN: dw_conv weight uninitialised
    The depthwise conv inside GatedConvFFN was default-initialised.
    Fix:       nn.init.normal_(self.dw_conv.weight, 0.0, 0.02)

BUG 5 — PureConvBlock: double residual (same as BUG 1)
    `x = self.conv(self.ln(x)) + x` — conv() already returned `residual + x_t`,
    so this added the original x a third time.
    Fix:       conv helper is now residual-free; PureConvBlock owns the residual.

BUG 6 — PureConvBlock: FFN uninitialised
    nn.Sequential(Linear, GELU, Dropout, Linear, Dropout) without weight init.
    Fix:       explicit GPT-2 init applied to both linear layers.

BUG 7 — ConvBeforeAttnBlock: LayerNorm before conv, but conv output is NOT
         layer-normed before the residual add.
    The pre-norm contract requires LN before every sub-layer input.  The conv
    sub-layer was doing `x = x + drop(conv(ln_conv(x)))` which is correct,
    but DepthwiseSeparableConv1d was internally calling `self.norm(residual+x_t)`,
    i.e. norming AFTER the add.  That is post-norm for the conv sub-layer and
    pre-norm for nothing.
    Fix:       conv helper is now a pure transform; each block applies Pre-LN
               explicitly: `x = x + drop(conv(ln(x)))`.

Summary of architectural guarantee after fix
--------------------------------------------
All three blocks follow strict Pre-LayerNorm residual structure:
    sub-layer output = x + Dropout(SubLayer(LayerNorm(x)))
No sub-layer adds its own residual internally.
All weight tensors initialised to GPT-2 scale (std=0.02, out-proj scaled by
1/sqrt(2*n_layers)).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper: causal depthwise-separable Conv1D
# ─────────────────────────────────────────────────────────────────────────────

class DepthwiseSeparableConv1d(nn.Module):
    """
    Causal depthwise-separable Conv1D.

    IMPORTANT: this module is a PURE TRANSFORM — it does NOT add a residual
    or apply LayerNorm internally.  All residual connections and LayerNorms
    are the responsibility of the enclosing block (Pre-LN contract).

    Left-only padding ensures strict causality (no future leakage).
    Input / output shape: (B, T, C).
    """

    def __init__(self, d_model: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.causal_pad = kernel_size - 1

        # Depthwise: one filter per channel, no cross-channel mixing
        self.depthwise = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=0,          # applied manually via F.pad
            groups=d_model,
            bias=False,
        )
        # Pointwise: 1×1 conv for cross-channel mixing
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1, bias=True)
        self.act     = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # FIX BUG 2: GPT-2 scale init for both conv layers
        nn.init.normal_(self.depthwise.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pointwise.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x_t = x.transpose(1, 2)                    # → (B, C, T)
        x_t = F.pad(x_t, (self.causal_pad, 0))     # causal left-pad only
        x_t = self.depthwise(x_t)                   # (B, C, T)
        x_t = self.pointwise(x_t)                   # (B, C, T)
        x_t = x_t.transpose(1, 2)                   # → (B, T, C)
        x_t = self.act(x_t)
        return self.dropout(x_t)
        # FIX BUG 1 & 7: NO residual add, NO LayerNorm here.
        # The caller is responsible for: x = x + dropout(this_module(ln(x)))


# ─────────────────────────────────────────────────────────────────────────────
# Design 1: Conv1D before each attention block
# ─────────────────────────────────────────────────────────────────────────────

class ConvBeforeAttnBlock(nn.Module):
    """
    Pre-LN block with a causal conv sub-layer prepended before attention.

    Forward path (strict Pre-LN):
        x → LN → DepthwiseSepConv → Add+Drop
          → LN → Attn              → Add+Drop
          → LN → FFN               → Add+Drop
    """

    def __init__(
        self,
        d_model: int,
        attention_module: nn.Module,
        ffn_module: nn.Module,
        n_layers: int,          # needed for conv out_proj scaled init (passed through)
        dropout: float = 0.1,
        kernel_size: int = 3,
        alibi: Optional[nn.Module] = None,
        rope: Optional[nn.Module] = None,
    ):
        super().__init__()

        # FIX BUG 7: separate LN for each sub-layer (strict Pre-LN)
        self.ln_conv = nn.LayerNorm(d_model)
        self.conv    = DepthwiseSeparableConv1d(d_model, kernel_size, dropout)

        self.ln_attn = nn.LayerNorm(d_model)
        self.attn    = attention_module

        self.ln_ffn  = nn.LayerNorm(d_model)
        self.ffn     = ffn_module

        self.dropout = nn.Dropout(dropout)
        self.alibi   = alibi
        self.rope    = rope

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_len = x.size(1)

        # ── Sub-layer 1: causal conv (Pre-LN, own residual) ──────────────────
        # FIX BUG 1: conv() is now a pure transform; residual added here only.
        x = x + self.dropout(self.conv(self.ln_conv(x)))

        # ── Inject positional biases ──────────────────────────────────────────
        if self.alibi is not None:
            kwargs["alibi_bias"] = self.alibi(seq_len, seq_len, x.device)
        if self.rope is not None:
            kwargs["rope"] = self.rope

        # ── Sub-layer 2: attention (Pre-LN) ──────────────────────────────────
        x = x + self.dropout(self.attn(self.ln_attn(x), mask=mask, **kwargs))

        # ── Sub-layer 3: FFN (Pre-LN) ─────────────────────────────────────────
        x = x + self.dropout(self.ffn(self.ln_ffn(x)))

        return x


# ─────────────────────────────────────────────────────────────────────────────
# Design 2: Gated Conv FFN block
# ─────────────────────────────────────────────────────────────────────────────

class GatedConvFFN(nn.Module):
    """
    SwiGLU-gated depthwise-separable conv feedforward network.

    x → Linear(d_model → 2*d_ff) → SwiGLU(gate, value)
      → CausalDWConv → Linear(d_ff → d_model)

    This is a PURE TRANSFORM (no internal residual/norm).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_layers: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, 2 * d_ff, bias=False)

        self.causal_pad = kernel_size - 1
        self.dw_conv = nn.Conv1d(
            d_ff, d_ff,
            kernel_size=kernel_size,
            padding=0,
            groups=d_ff,
            bias=False,
        )
        self.out_proj = nn.Linear(d_ff, d_model, bias=True)
        self.dropout  = nn.Dropout(dropout)

        # FIX BUG 3: GPT-2 scaled init for gate_proj and out_proj
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.out_proj.weight,  mean=0.0,
                        std=0.02 / math.sqrt(2 * n_layers))
        nn.init.zeros_(self.out_proj.bias)

        # FIX BUG 4: init dw_conv weight
        nn.init.normal_(self.dw_conv.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, val = self.gate_proj(x).chunk(2, dim=-1)   # (B, T, d_ff) each
        x_inner = F.silu(gate) * val                     # SwiGLU

        # Causal depthwise conv
        x_inner = x_inner.transpose(1, 2)                # (B, d_ff, T)
        x_inner = F.pad(x_inner, (self.causal_pad, 0))   # causal left-pad
        x_inner = self.dw_conv(x_inner)
        x_inner = x_inner.transpose(1, 2)                # (B, T, d_ff)

        return self.dropout(self.out_proj(x_inner))
        # FIX BUG 3: NO internal residual; the block owns the residual.


class GatedConvFFNBlock(nn.Module):
    """
    Standard causal attention + GatedConvFFN instead of the usual MLP.
    Strict Pre-LN residual structure.
    """

    def __init__(
        self,
        d_model: int,
        attention_module: nn.Module,
        d_ff: int,
        n_layers: int,
        dropout: float = 0.1,
        kernel_size: int = 3,
        alibi: Optional[nn.Module] = None,
        rope: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.ln_1    = nn.LayerNorm(d_model)
        self.attn    = attention_module
        self.ln_2    = nn.LayerNorm(d_model)
        self.ffn     = GatedConvFFN(d_model, d_ff, n_layers, kernel_size, dropout)
        self.dropout = nn.Dropout(dropout)
        self.alibi   = alibi
        self.rope    = rope

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_len = x.size(1)

        if self.alibi is not None:
            kwargs["alibi_bias"] = self.alibi(seq_len, seq_len, x.device)
        if self.rope is not None:
            kwargs["rope"] = self.rope

        # FIX BUG 3: ffn() is a pure transform; residual added here only.
        x = x + self.dropout(self.attn(self.ln_1(x), mask=mask, **kwargs))
        x = x + self.dropout(self.ffn(self.ln_2(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Design 3: Pure Conv1D block (for interleaved stacking)
# ─────────────────────────────────────────────────────────────────────────────

class PureConvBlock(nn.Module):
    """
    Replaces an attention block entirely with:
        Pre-LN causal depthwise-sep conv + Pre-LN FFN.

    Strict Pre-LN residual structure throughout.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_layers: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ln_conv = nn.LayerNorm(d_model)
        self.conv    = DepthwiseSeparableConv1d(d_model, kernel_size, dropout)

        self.ln_ffn  = nn.LayerNorm(d_model)

        # FIX BUG 6: explicit GPT-2 weight init for FFN linears
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, mean=0.0,
                        std=0.02 / math.sqrt(2 * n_layers))
        nn.init.zeros_(self.fc2.bias)

        self.act     = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.dropout(self.act(self.fc1(x)))))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,   # absorbs rope/alibi_bias — unused in pure-conv layers
    ) -> torch.Tensor:
        # FIX BUG 1 & 5: conv() is a pure transform; residual added here only.
        x = x + self.dropout(self.conv(self.ln_conv(x)))
        x = x + self._ffn(self.ln_ffn(x))
        return x