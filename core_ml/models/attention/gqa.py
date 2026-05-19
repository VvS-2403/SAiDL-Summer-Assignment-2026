import torch
import torch.nn as nn
import math
from einops import rearrange, repeat

class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA).
    Bridges the gap between Multi-Head Attention (MHA) and Multi-Query Attention (MQA)
    by dividing query heads into groups, where each group shares a single Key and Value head.
    Significantly reduces KV-cache memory during autoregressive inference.
    """
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int, dropout: float = 0.1, is_causal: bool = True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.d_k = d_model // num_heads
        self.is_causal = is_causal
        
        # In GQA, Q has more heads than K and V
        self.q_proj = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, **kwargs) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # 1. Independent Projections for Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 2. Reshape into heads
        # q shape: (batch, num_heads, seq_len, d_k)
        # k, v shape: (batch, num_kv_heads, seq_len, d_k)
        q = rearrange(q, 'b s (h d) -> b h s d', h=self.num_heads)
        k = rearrange(k, 'b s (g d) -> b g s d', g=self.num_kv_heads)
        v = rearrange(v, 'b s (g d) -> b g s d', g=self.num_kv_heads)
        
        # 3. Broadcast (Repeat) K and V to match the number of Query heads
        # We repeat each KV head 'num_groups' times adjacent to each other
        # k, v shape becomes: (batch, num_heads, seq_len, d_k)
        k = repeat(k, 'b g s d -> b (g r) s d', r=self.num_groups)
        v = repeat(v, 'b g s d -> b (g r) s d', r=self.num_groups)
        
        # 4. Standard Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if self.is_causal and mask is None:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1, seq_len, seq_len)
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        elif mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        
        # 5. Concatenate and Project
        context = rearrange(context, 'b h s d -> b s (h d)')
        output = self.resid_dropout(self.out_proj(context))
        
        return output