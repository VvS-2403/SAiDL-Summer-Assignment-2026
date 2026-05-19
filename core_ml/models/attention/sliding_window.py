import torch
import torch.nn as nn
import math
from einops import rearrange

class SlidingWindowAttention(nn.Module):
    """
    Sliding Window (Local) Multi-Head Attention.
    Restricts attention to a fixed-size window of recent tokens to reduce memory complexity
    from O(N^2) to O(N * W).
    """
    def __init__(self, d_model: int, num_heads: int, window_size: int = 256, dropout: float = 0.1, is_causal: bool = True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.window_size = window_size
        self.is_causal = is_causal
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, **kwargs) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # 1. Project and split Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, 'b s (three h d) -> three b h s d', three=3, h=self.num_heads)
        
        # 2. Compute raw attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 3. Create the Sliding Window Mask
        # We start with a matrix of indices
        row_indices = torch.arange(seq_len, device=x.device).unsqueeze(1)
        col_indices = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        # Calculate distance between query (row) and key (col)
        distance = row_indices - col_indices
        
        # Condition 1: Must not look into the future (Causal)
        # Condition 2: Must not look further back than window_size
        window_mask = (distance >= 0) & (distance < self.window_size)
        
        # Invert the boolean mask: True where we want to BLOCK attention
        block_mask = ~window_mask
        
        # Apply the window mask
        scores = scores.masked_fill(block_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply external padding mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        # 4. Standard Attention processing
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        context = rearrange(context, 'b h s d -> b s (h d)')
        output = self.resid_dropout(self.out_proj(context))
        
        return output