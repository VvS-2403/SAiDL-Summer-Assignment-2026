import torch
import torch.nn as nn
import math

class ALiBiPositionalBias(nn.Module):
    """
    Attention with Linear Biases (ALiBi).
    Instead of adding positional embeddings to the tokens at the bottom of the network,
    ALiBi adds a static, distance-proportional bias directly to the attention scores
    (before the Softmax) in every single layer.
    """
    def __init__(self, num_heads: int):
        """
        Args:
            num_heads (int): The number of attention heads. ALiBi requires unique slopes per head.
        """
        super().__init__()
        self.num_heads = num_heads
        
        # Calculate the geometric sequence of slopes for the attention heads
        slopes = self._get_slopes(num_heads)
        
        # Register slopes as a buffer so it moves to the GPU automatically
        # Shape: (num_heads, 1, 1) for easy broadcasting
        self.register_buffer("slopes", torch.tensor(slopes, dtype=torch.float32).view(num_heads, 1, 1))

    def _get_slopes(self, n: int) -> list[float]:
        """
        Mathematically derives the ALiBi slopes.
        For 8 heads, the slopes are: 1/2^1, 1/2^2, ..., 1/2^8.
        This function robustly handles head counts that are not perfect powers of 2.
        """
        def get_slopes_power_of_2(n):
            start = (2 ** (-8 / n))
            return [start * (start ** i) for i in range(n)]
        
        # If perfect power of 2 (e.g., 8, 16, 32)
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            # Fallback for non-power-of-2 heads (e.g., 12 heads)
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + \
                   self._get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

    def forward(self, q_seq_len: int, k_seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generates the ALiBi bias matrix on the fly.
        
        Args:
            q_seq_len (int): Sequence length of the queries.
            k_seq_len (int): Sequence length of the keys.
            device (torch.device): The device of the current attention tensors.
            
        Returns:
            torch.Tensor: The ALiBi bias matrix of shape (1, num_heads, q_seq_len, k_seq_len)
        """
        # Create column and row indices
        q_pos = torch.arange(q_seq_len, device=device).view(-1, 1)
        k_pos = torch.arange(k_seq_len, device=device).view(1, -1)
        
        # Calculate relative distance: (Key Position) - (Query Position)
        # If Query is looking at a past Key, this value will be negative.
        # Shape: (q_seq_len, k_seq_len)
        distance = k_pos - q_pos
        
        # Expand dimensions to match standard attention score shapes
        # Shape: (1, 1, q_seq_len, k_seq_len)
        distance = distance.unsqueeze(0).unsqueeze(0)
        
        # Multiply distance by the head-specific slopes
        # Shape: (1, num_heads, q_seq_len, k_seq_len)
        alibi_bias = distance * self.slopes.unsqueeze(0)
        
        return alibi_bias