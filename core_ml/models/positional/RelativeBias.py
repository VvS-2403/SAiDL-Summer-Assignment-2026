import torch
import torch.nn as nn
import math


class RelativePositionalBias(nn.Module):
    """
    Learned Relative Positional Bias (Shaw et al., 2018).

    Instead of adding absolute position encodings to embeddings, this module
    adds a learned scalar bias to each (query_pos, key_pos) pair in the
    attention score matrix.  The bias depends only on the *clipped* signed
    distance (key_pos - query_pos), so the model learns a single embedding
    table of size (2 * max_relative_distance + 1) and looks up the right
    entry for every pair.

    Usage — identical to ALiBi:
        • Pass nn.Identity() as positional_encoding to the Transformer shell.
        • Instantiate one RelativePositionalBias and inject it into every
          TransformerBlock via the `alibi` kwarg slot (the block calls it the
          same way and adds the result to attention scores).
    """

    def __init__(self, num_heads: int, max_relative_distance: int = 128):
        """
        Args:
            num_heads (int): Number of attention heads.
            max_relative_distance (int): Distances beyond this value are
                clipped.  Keeps the table size bounded for long contexts.
        """
        super().__init__()
        self.num_heads = num_heads
        self.max_dist = max_relative_distance

        # Table size: negative distances + zero + positive distances
        # Index 0 = most-negative clip, index 2*max_dist = most-positive clip
        num_buckets = 2 * max_relative_distance + 1
        self.relative_bias_table = nn.Embedding(num_buckets, num_heads)

        # Small normal init — same scale used for other learned embeddings
        nn.init.normal_(self.relative_bias_table.weight, mean=0.0, std=0.02)

    def _clip_distance(self, distance: torch.Tensor) -> torch.Tensor:
        """Clips distances to [-max_dist, +max_dist] then shifts to [0, 2*max_dist]."""
        distance = distance.clamp(-self.max_dist, self.max_dist)
        return distance + self.max_dist  # shift so index 0 = most negative

    def forward(self, q_seq_len: int, k_seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generates the relative positional bias matrix.

        Args:
            q_seq_len: Number of query positions.
            k_seq_len: Number of key positions.
            device:    Device of the attention tensors.

        Returns:
            Tensor of shape (1, num_heads, q_seq_len, k_seq_len) to be added
            to raw attention scores before softmax.
        """
        q_pos = torch.arange(q_seq_len, device=device).unsqueeze(1)   # (q, 1)
        k_pos = torch.arange(k_seq_len, device=device).unsqueeze(0)   # (1, k)

        # Signed distance: positive = key is ahead, negative = key is behind
        distance = k_pos - q_pos                                       # (q, k)
        bucket_ids = self._clip_distance(distance)                     # (q, k)

        # Look up learned biases: (q, k, num_heads)
        biases = self.relative_bias_table(bucket_ids)

        # Rearrange to (1, num_heads, q, k) to broadcast over batch dimension
        biases = biases.permute(2, 0, 1).unsqueeze(0)                 # (1, H, q, k)
        return biases
