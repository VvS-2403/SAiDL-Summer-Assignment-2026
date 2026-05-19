import torch
import torch.nn as nn
import math
from einops import rearrange

class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Self-Attention (MHA) as defined in "Attention Is All You Need".
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, is_causal: bool = True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.is_causal = is_causal
        
        # Combined linear projection for Query, Key, and Value
        # Output shape will be 3 * d_model to split later
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        
        # Final output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional custom mask. If None and is_causal is True, a causal mask is generated.
        """
        batch_size, seq_len, _ = x.size()
        
        # 1. Project Q, K, V simultaneously
        # Shape: (batch_size, seq_len, 3 * d_model)
        qkv = self.qkv_proj(x)
        
        # 2. Split into Q, K, V and rearrange for multi-head processing
        # Using einops: splits the last dimension into 3, extracts heads, and swaps seq_len with num_heads
        # Resulting shapes for q, k, v: (batch_size, num_heads, seq_len, d_k)
        q, k, v = rearrange(qkv, 'b s (three h d) -> three b h s d', three=3, h=self.num_heads)
        
        # 3. Compute Attention Scores: (Q * K^T) / sqrt(d_k)
        # k.transpose(-2, -1) flips the last two dimensions (seq_len and d_k) for matrix multiplication
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 4. Apply Masks
        if self.is_causal and mask is None:
            # Create a lower triangular mask on the fly
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1, seq_len, seq_len)
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        elif mask is not None:
            # Apply external mask (e.g., padding mask)
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        # 5. Softmax to get probabilities
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 6. Apply weights to Values
        # Shape: (batch_size, num_heads, seq_len, d_k)
        context = torch.matmul(attn_weights, v)
        
        # 7. Concatenate heads back together
        # Reverses the earlier einops rearrangement: (batch_size, seq_len, d_model)
        context = rearrange(context, 'b h s d -> b s (h d)')
        
        # 8. Final linear projection
        output = self.resid_dropout(self.out_proj(context))
        
        return output