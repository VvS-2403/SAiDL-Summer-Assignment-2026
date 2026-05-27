"""
Convolution + Attention Hybrid Blocks.

Three designs implemented here:

1. ConvPreAttnBlock  — Conv1D applied to the residual stream BEFORE each attention sub-layer.
   (Local n-gram feature extraction before global attention.)

2. InterleavedConvAttnBlock — Alternating: odd layers are pure Conv, even layers are pure Attn.
   Drop this into TransformerBlock's attention slot or use as a standalone block stack.

3. GatedConvFFN — Replaces the standard FFN with a Gated Depthwise-Separable 1D Conv FFN.
   Inspired by Conformer (Gulati et al., 2020) and Gated Linear Units.

All blocks accept the same (x, mask=None, **kwargs) signature as the standard
TransformerBlock so they slot directly into the existing Transformer shell.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─── helpers ──────────────────────────────────────────────────────────────────

class CausalDepthwiseConv1d(nn.Module):
    """
    Causal depthwise conv: no future leakage, O(N * kernel_size) complexity.
    We left-pad by (kernel_size - 1) before convolving so output length = input length.
    """
    def __init__(self, channels: int, kernel_size: int = 3, groups: int = None):
        super().__init__()
        groups = groups or channels  # fully depthwise by default
        self.pad  = kernel_size - 1
        self.conv = nn.Conv1d(channels, channels, kernel_size, groups=groups, bias=False)
        nn.init.normal_(self.conv.weight, 0.0, 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) → conv expects (B, C, T)
        x = x.transpose(1, 2)
        x = F.pad(x, (self.pad, 0))   # causal left-pad
        x = self.conv(x)
        return x.transpose(1, 2)       # back to (B, T, C)


class PointwiseConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=True)
        nn.init.normal_(self.conv.weight, 0.0, 0.02)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


# ─── Design 1: Conv before Attention ──────────────────────────────────────────

class ConvPreAttnBlock(nn.Module):
    """
    Pre-Norm block:  x → LN → DWConv → Add → LN → Attn → Add → LN → FFN → Add

    The depthwise conv acts as a cheap local feature extractor that primes Q/K/V.
    """
    def __init__(
        self,
        d_model: int,
        attention_module: nn.Module,
        ffn_module: nn.Module,
        kernel_size: int = 3,
        dropout: float = 0.1,
        alibi=None,
    ):
        super().__init__()
        self.ln_conv = nn.LayerNorm(d_model)
        self.conv    = CausalDepthwiseConv1d(d_model, kernel_size)
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn    = attention_module
        self.ln_ffn  = nn.LayerNorm(d_model)
        self.ffn     = ffn_module
        self.drop    = nn.Dropout(dropout)
        self.alibi   = alibi

    def forward(self, x: torch.Tensor, mask=None, **kwargs) -> torch.Tensor:
        # 1. Local conv sub-layer
        x = x + self.drop(self.conv(self.ln_conv(x)))

        # 2. Attention sub-layer
        if self.alibi is not None:
            kwargs['alibi_bias'] = self.alibi(x.size(1), x.size(1), x.device)
        x = x + self.drop(self.attn(self.ln_attn(x), mask=mask, **kwargs))

        # 3. FFN sub-layer
        x = x + self.drop(self.ffn(self.ln_ffn(x)))
        return x


# ─── Design 2: Gated Conv FFN (replaces standard FFN) ─────────────────────────

class GatedConvFFN(nn.Module):
    """
    Gated Depthwise-Separable Convolutional FFN.

    Architecture:
        x → LN → PW_expand → GELU gate | linear gate → GLU → DW_conv → PW_contract → Dropout

    Replaces the standard Linear→GELU→Linear FFN.
    The depthwise conv provides local context; GLU provides selective gating.
    Parameter count is comparable to the standard FFN (slightly fewer ops).
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
        self.expand   = PointwiseConv1d(d_model, d_ff * 2)   # factor of 2 for GLU
        self.dw_conv  = CausalDepthwiseConv1d(d_ff, kernel_size)
        self.contract = PointwiseConv1d(d_ff, d_model)
        self.dropout  = nn.Dropout(dropout)

        # Scaled init on the output projection
        nn.init.normal_(self.contract.conv.weight, 0.0, 0.02 / math.sqrt(2 * n_layers))

        # store for logging compatibility with FeedForward interface
        self.inter_mean = 0.0
        self.inter_std  = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expand and split for Gated Linear Unit
        gate, linear = self.expand(x).chunk(2, dim=-1)  # each (B, T, d_ff)
        x_inner = F.gelu(gate) * linear                 # GLU
        x_inner = self.dw_conv(x_inner)                 # local conv

        if self.training:
            with torch.no_grad():
                self.inter_mean = x_inner.mean().item()
                self.inter_std  = x_inner.std().item()

        return self.dropout(self.contract(x_inner))


# ─── Design 3: Interleaved Conv-Attn stack helper ─────────────────────────────

class InterleavedConvAttnBlock(nn.Module):
    """
    Interleaved block: even layer_idx → attention, odd layer_idx → conv.
    The 'attention_module' slot can be None if this is a conv-only layer.

    Usage: pass layer_idx at construction time; the block decides which sub-layer
    is its primary operation.  FFN is always present.
    """
    def __init__(
        self,
        d_model: int,
        attention_module: nn.Module,   # can be None for conv-only layers
        ffn_module: nn.Module,
        layer_idx: int,
        kernel_size: int = 7,
        dropout: float = 0.1,
        alibi=None,
    ):
        super().__init__()
        self.use_attn   = (layer_idx % 2 == 0)
        self.ln_1       = nn.LayerNorm(d_model)
        self.attn       = attention_module
        self.conv       = CausalDepthwiseConv1d(d_model, kernel_size) if not self.use_attn else None
        self.ln_2       = nn.LayerNorm(d_model)
        self.ffn        = ffn_module
        self.drop       = nn.Dropout(dropout)
        self.alibi      = alibi

    def forward(self, x: torch.Tensor, mask=None, **kwargs) -> torch.Tensor:
        # Sub-layer 1: attn or conv
        if self.use_attn:
            if self.alibi is not None:
                kwargs['alibi_bias'] = self.alibi(x.size(1), x.size(1), x.device)
            x = x + self.drop(self.attn(self.ln_1(x), mask=mask, **kwargs))
        else:
            x = x + self.drop(self.conv(self.ln_1(x)))

        # Sub-layer 2: FFN always
        x = x + self.drop(self.ffn(self.ln_2(x)))
        return x