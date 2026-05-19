import torch
import torch.nn as nn
import math
from einops import rearrange

class SparseAttention(nn.Module):
    """
    Simulated Fixed-Pattern Sparse Multi-Head Attention.
    Combines a local window with a strided global attention pattern to achieve sub-quadratic
    memory complexity while maintaining long-range communication.
    """
    def __init__(self, d_model: int, num_heads: int, local_window: int = 64, stride: int = 64, dropout: float = 0.1, is_causal: bool = True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.local_window = local_window
        self.stride = stride
        self.is_causal = is_causal
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, **kwargs) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # 1. Project Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, 'b s (three h d) -> three b h s d', three=3, h=self.num_heads)
        
        # 2. Raw Dot Products
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 3. Build the Sparse Mask
        row_indices = torch.arange(seq_len, device=x.device).unsqueeze(1)
        col_indices = torch.arange(seq_len, device=x.device).unsqueeze(0)
        distance = row_indices - col_indices
        
        # Condition A: Local Neighborhood (Sliding Window)
        is_local = (distance >= 0) & (distance < self.local_window)
        
        # Condition B: Strided Global Tokens (Look back at every k-th token)
        is_strided = (distance >= 0) & (distance % self.stride == 0)
        
        # A token is allowed to attend if it's in the local window OR it aligns with the stride
        sparse_mask = is_local | is_strided
        
        # Apply Causality: Cannot look into the future
        if self.is_causal:
            sparse_mask = sparse_mask & (distance >= 0)
            
        # Invert the boolean mask to find blocks that must be zeroed out
        block_mask = ~sparse_mask
        
        # 4. Apply Masking
        scores = scores.masked_fill(block_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        # 5. Attention Weights and Context
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        context = rearrange(context, 'b h s d -> b s (h d)')
        output = self.resid_dropout(self.out_proj(context))
        
        return output