import torch
import torch.nn as nn
import math
from einops import rearrange

class ReLUAttention(nn.Module):
    """
    ReLU Attention Mechanism.
    Replaces the computationally expensive Softmax operation with a simple ReLU activation.
    This introduces true sparsity into the attention matrix (values drop to exactly 0)
    and removes the exponential bottleneck.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, is_causal: bool = True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
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
        
        # 3. Apply Causality Masking
        # We use -inf here so that after ReLU it becomes strictly 0.
        if self.is_causal and mask is None:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1, seq_len, seq_len)
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        elif mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        # 4. The Architectural Swap: ReLU instead of Softmax
        # Any negative dot product becomes exactly 0.0
        attn_weights = torch.relu(scores)
        
        # In ReLU attention, the sum of weights is no longer naturally 1.0.
        # To prevent massive variance shifts, we normalize by the sequence length.
        attn_weights = attn_weights / seq_len
        
        attn_weights = self.attn_dropout(attn_weights)
        
        # 5. Apply weights to Values
        context = torch.matmul(attn_weights, v)
        
        # 6. Concatenate and Project
        context = rearrange(context, 'b h s d -> b s (h d)')
        output = self.resid_dropout(self.out_proj(context))
        
        return output