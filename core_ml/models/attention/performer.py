import torch
import torch.nn as nn
import math
from einops import rearrange

class PerformerAttention(nn.Module):
    """
    Simplified Performer Attention (FAVOR+ approximation).
    Approximates the standard Softmax attention kernel using random feature maps,
    changing the time/memory complexity from O(N^2) to O(N * d).
    """
    def __init__(self, d_model: int, num_heads: int, num_random_features: int = 256, dropout: float = 0.1, is_causal: bool = False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.num_features = num_random_features
        self.is_causal = is_causal
        
        # Causal Performer requires a prefix-sum (cumulative sum) approach which is
        # complex to vectorize cleanly without custom CUDA kernels.
        # This baseline implementation focuses on the bidirectional (non-causal) approximation.
        if is_causal:
            raise NotImplementedError("Causal masking for Performer requires custom parallel prefix-sum kernels not included in this baseline.")
            
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Random projection matrix (fixed during training, hence buffer)
        # Shape: (num_heads, d_k, num_random_features)
        random_matrix = torch.randn(num_heads, self.d_k, self.num_features) / math.sqrt(self.d_k)
        self.register_buffer("random_matrix", random_matrix)
        
        self.resid_dropout = nn.Dropout(dropout)

    def kernel_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a positive random feature approximation to the input tensor.
        Instead of the exact exponential MAC of FAVOR+, we use a numerically stable
        ReLU-based positive random feature projection as a simplified kernel trick.
        """
        # x shape: (batch, heads, seq, d_k)
        # random_matrix shape: (heads, d_k, num_features)
        # Einsum multiplication: project d_k to num_features
        projection = torch.einsum('b h s d, h d f -> b h s f', x, self.random_matrix)
        
        # Apply non-linearity to ensure positive features (required for probability estimation)
        # Added small epsilon to prevent division by zero later
        return torch.relu(projection) + 1e-6

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, **kwargs) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # 1. Project Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, 'b s (three h d) -> three b h s d', three=3, h=self.num_heads)
        
        # 2. Apply the Kernel Trick (Map Q and K to positive random features)
        # q_prime, k_prime shape: (batch_size, num_heads, seq_len, num_features)
        q_prime = self.kernel_feature_map(q)
        k_prime = self.kernel_feature_map(k)
        
        # 3. The Associativity Trick
        # Standard: (Q * K^T) * V  -> O(N^2)
        # Performer: Q * (K^T * V) -> O(N * d)
        
        # First, multiply K^T and V
        # k_prime^T shape: (batch, heads, num_features, seq_len)
        # v shape: (batch, heads, seq_len, d_k)
        # k_v_product shape: (batch, heads, num_features, d_k)
        k_v_product = torch.einsum('b h s f, b h s d -> b h f d', k_prime, v)
        
        # Second, multiply Q by the resulting matrix
        # context shape: (batch, heads, seq_len, d_k)
        context = torch.einsum('b h s f, b h f d -> b h s d', q_prime, k_v_product)
        
        # 4. Normalization (Denominator of the Softmax approximation)
        # We must divide by the sum of the keys to properly normalize the probabilities
        k_sum = k_prime.sum(dim=-2) # Shape: (batch, heads, num_features)
        denominator = torch.einsum('b h s f, b h f -> b h s', q_prime, k_sum).unsqueeze(-1)
        
        context = context / denominator
        
        # 5. Concatenate and Project
        context = rearrange(context, 'b h s d -> b s (h d)')
        output = self.resid_dropout(self.out_proj(context))
        
        return output