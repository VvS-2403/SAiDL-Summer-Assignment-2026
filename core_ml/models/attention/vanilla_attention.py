import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Self-Attention with GPT-2 scaled initialization
    and support for ALiBi positional biases.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.1,
        is_causal: bool = True
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be exactly divisible by n_heads."

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.is_causal = is_causal

        # 1. Combined Q, K, V projection for maximum throughput
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        
        # 2. Output projection back to residual stream
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # ------------------------------------------------------------------
        # GPT-2 SCALED WEIGHT INITIALIZATION
        # ------------------------------------------------------------------
        # Standard normal initialization for the QKV matrices
        nn.init.normal_(self.qkv_proj.weight, mean=0.0, std=0.02)
        if self.qkv_proj.bias is not None:
            nn.init.zeros_(self.qkv_proj.bias)

        # Scaled initialization for the output projection to prevent variance 
        # explosion deep in the residual stream (fixes vanishing gradients).
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, alibi: torch.Tensor = None, **kwargs):
        """
        Args:
            x (torch.Tensor): Input sequence of shape (B, T, C).
            mask (torch.Tensor, optional): Boolean causal mask.
            alibi (torch.Tensor, optional): ALiBi distance penalty bias.
        """
        B, T, C = x.size()

        # 1. Project and split into Query, Key, Value
        qkv = self.qkv_proj(x) # Shape: (B, T, 3 * d_model)
        q, k, v = qkv.split(self.d_model, dim=2)

        # Reshape for multi-head attention: (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # 2. Compute Attention Scores: (Q @ K^T) / sqrt(d_head)
        # Shape: (B, n_heads, T, d_head) @ (B, n_heads, d_head, T) -> (B, n_heads, T, T)
        scores = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(self.d_head))

        # 3. Inject ALiBi Bias (If present)
        if alibi is not None:
            # ALiBi modifies the attention scores before the softmax
            scores = scores + alibi

        # 4. Apply Causal Masking
        if self.is_causal:
            if mask is None:
                # Generate a standard causal mask on the fly if not provided
                mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device)).view(1, 1, T, T)
            
            # Fill the masked positions with -inf so they softmax to 0
            scores = scores.masked_fill(~mask, float('-inf'))

        # 5. Softmax and Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 6. Compute weighted sum of values
        # Shape: (B, n_heads, T, T) @ (B, n_heads, T, d_head) -> (B, n_heads, T, d_head)
        out = attn_weights @ v

        # 7. Re-assemble all head outputs
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # 8. Project back to the residual stream
        return self.resid_dropout(self.out_proj(out))
