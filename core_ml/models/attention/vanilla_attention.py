import torch
import torch.nn as nn
import math
from einops import rearrange


class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Self-Attention (MHA) as defined in "Attention Is All You Need".
    Supports an optional ALiBi positional bias passed in via **kwargs.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        is_causal: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self.is_causal = is_causal

        # Combined linear projection for Query, Key, and Value
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)

        # Final output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout  = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x        : (batch_size, seq_len, d_model)
            mask     : Optional custom mask.
            **kwargs : May contain `alibi_bias` of shape
                       (1, num_heads, seq_len, seq_len) when ALiBi is active.
        """
        batch_size, seq_len, _ = x.size()

        # 1. Project Q, K, V simultaneously
        qkv = self.qkv_proj(x)

        # 2. Split into Q, K, V and rearrange for multi-head processing
        # Shapes: (batch_size, num_heads, seq_len, d_k)
        q, k, v = rearrange(qkv, 'b s (three h d) -> three b h s d', three=3, h=self.num_heads)

        # 3. Scaled dot-product scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 4a. Add ALiBi bias BEFORE the causal mask so -inf positions stay -inf
        #     alibi_bias shape: (1, num_heads, seq_len, seq_len) — broadcasts over batch
        alibi_bias = kwargs.get("alibi_bias", None)
        if alibi_bias is not None:
            scores = scores + alibi_bias

        # 4b. Apply causal or padding mask
        if self.is_causal and mask is None:
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=x.device)
            ).view(1, 1, seq_len, seq_len)
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        elif mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 5. Softmax → probabilities
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 6. Weighted sum of Values
        context = torch.matmul(attn_weights, v)

        # 7. Concatenate heads
        context = rearrange(context, 'b h s d -> b s (h d)')

        # 8. Final linear projection
        output = self.resid_dropout(self.out_proj(context))

        return output
