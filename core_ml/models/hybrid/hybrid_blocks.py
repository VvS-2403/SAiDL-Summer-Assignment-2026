
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper: depthwise-separable Conv1D
# ─────────────────────────────────────────────────────────────────────────────

class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise Conv1D (groups=channels) + pointwise projection.
    Input/output shape: (B, T, C).
    """
    def __init__(self, d_model: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=padding, groups=d_model, bias=False,
        )
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1, bias=True)
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act     = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_t = x.transpose(1, 2)          # (B, C, T)
        x_t = self.depthwise(x_t)
        x_t = self.pointwise(x_t)
        x_t = x_t.transpose(1, 2)        # (B, T, C)
        x_t = self.act(x_t)
        x_t = self.dropout(x_t)
        return self.norm(residual + x_t)


# ─────────────────────────────────────────────────────────────────────────────
# Design 1: Conv1D before each attention block
# ─────────────────────────────────────────────────────────────────────────────

class ConvBeforeAttnBlock(nn.Module):
    """
    Prepends a depthwise-separable Conv1D sub-layer before attention.

    Forward path:
        x → DepthwiseSepConv1d → [attn + ffn with Pre-LN] → x_out
    """

    def __init__(
        self,
        d_model: int,
        attention_module: nn.Module,
        ffn_module: nn.Module,
        dropout: float = 0.1,
        kernel_size: int = 3,
        alibi: Optional[nn.Module] = None,
        rope:  Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv    = DepthwiseSeparableConv1d(d_model, kernel_size, dropout)
        self.ln_1    = nn.LayerNorm(d_model)
        self.attn    = attention_module
        self.ln_2    = nn.LayerNorm(d_model)
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
        # 1. Local n-gram context via conv
        x = self.conv(x)

        # 2. Inject positional biases into kwargs for attention
        seq_len = x.size(1)
        if self.alibi is not None:
            kwargs["alibi_bias"] = self.alibi(seq_len, seq_len, x.device)
        if self.rope is not None:
            kwargs["rope"] = self.rope

        # 3. Attention sub-layer (Pre-LN)
        x = x + self.dropout(self.attn(self.ln_1(x), mask=mask, **kwargs))

        # 4. FFN sub-layer (Pre-LN)
        x = x + self.dropout(self.ffn(self.ln_2(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Design 2: Gated Convolutional FFN block
# ─────────────────────────────────────────────────────────────────────────────

class GatedConvFFN(nn.Module):
    """
    GLU-style gated depthwise-separable conv feedforward network.

    x → Linear(d_model → 2*d_ff) → sigmoid(gate) * GELU(value)
      → DepthwiseConv1d → Linear(d_ff → d_model)
    """

    def __init__(self, d_model: int, d_ff: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, 2 * d_ff, bias=False)
        padding = (kernel_size - 1) // 2
        self.dw_conv  = nn.Conv1d(d_ff, d_ff, kernel_size=kernel_size,
                                   padding=padding, groups=d_ff, bias=False)
        self.out_proj = nn.Linear(d_ff, d_model, bias=True)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, val = self.gate_proj(x).chunk(2, dim=-1)   # (B, T, d_ff) each
        x_inner = torch.sigmoid(gate) * F.gelu(val)
        x_inner = x_inner.transpose(1, 2)                # (B, d_ff, T)
        x_inner = self.dw_conv(x_inner)
        x_inner = x_inner.transpose(1, 2)                # (B, T, d_ff)
        return self.dropout(self.out_proj(x_inner))


class GatedConvFFNBlock(nn.Module):
    """
    Standard causal attention + GatedConvFFN instead of the usual MLP.
    """

    def __init__(
        self,
        d_model: int,
        attention_module: nn.Module,
        d_ff: int,
        dropout: float = 0.1,
        kernel_size: int = 3,
        alibi: Optional[nn.Module] = None,
        rope:  Optional[nn.Module] = None,
    ):
        super().__init__()
        self.ln_1    = nn.LayerNorm(d_model)
        self.attn    = attention_module
        self.ln_2    = nn.LayerNorm(d_model)
        self.ffn     = GatedConvFFN(d_model, d_ff, kernel_size, dropout)
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

        x = x + self.dropout(self.attn(self.ln_1(x), mask=mask, **kwargs))
        x = x + self.dropout(self.ffn(self.ln_2(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Design 3: Pure Conv1D block (for interleaved stacking)
# ─────────────────────────────────────────────────────────────────────────────

class PureConvBlock(nn.Module):
    """
    Replaces an entire attention block with depthwise-separable Conv1D + FFN.
    Used at every even layer index in the interleaved hybrid design.
    """

    def __init__(self, d_model: int, d_ff: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.ln   = nn.LayerNorm(d_model)
        self.conv = DepthwiseSeparableConv1d(d_model, kernel_size, dropout)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,                             # absorbs rope/alibi_bias — not used here
    ) -> torch.Tensor:
        x = self.conv(self.ln(x)) + x
        x = x + self.ffn(self.ln2(x))
        return x