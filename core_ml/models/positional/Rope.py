import torch
import torch.nn as nn

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Splits the last dimension of the tensor in half and rotates it.
    This mimics the mathematical effect of multiplying by a complex number i.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) as introduced in the RoFormer paper.
    Used in LLaMA, Mistral, and most state-of-the-art architectures.
    Note: RoPE is applied directly to Queries (Q) and Keys (K) inside the attention block,
    NOT to the flat token embeddings at the bottom of the model.
    """
    def __init__(self, d_model: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        
        # Calculate the inverse frequencies (theta_i)
        # Shape: (d_model // 2)
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Cache the maximum allowed sequence length
        self.max_seq_len_cached = max_seq_len
        self._set_cos_sin_cache(seq_len=max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int):
        """
        Pre-computes the cosine and sine matrices up to the maximum sequence length.
        """
        self.max_seq_len_cached = seq_len
        
        # t represents the position indices [0, 1, 2, ..., seq_len - 1]
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        
        # Outer product: position * inverse_frequency
        # Shape: (seq_len, d_model // 2)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # Duplicate frequencies to match the full d_model dimension (for rotate_half)
        # Shape: (seq_len, d_model)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Create 4D tensors to easily broadcast against (batch, num_heads, seq_len, d_k)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: Query tensor of shape (batch, num_heads, seq_len, d_k)
            k: Key tensor of shape (batch, num_heads, seq_len, d_k)
        Returns:
            Tuple of rotated (q, k) tensors.
        """
        seq_len = q.shape[2]
        
        # Dynamically extend the cache if the sequence exceeds our pre-computed length
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len)

        # Slice the cached positional matrices up to the current sequence length
        cos = self.cos_cached[:, :, :seq_len, ...].to(dtype=q.dtype, device=q.device)
        sin = self.sin_cached[:, :, :seq_len, ...].to(dtype=q.dtype, device=q.device)

        # Apply the rotary transformation
        q_rotated = (q * cos) + (rotate_half(q) * sin)
        k_rotated = (k * cos) + (rotate_half(k) * sin)

        return q_rotated, k_rotated