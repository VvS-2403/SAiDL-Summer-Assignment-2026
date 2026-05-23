import torch
import torch.nn as nn

class Transformer(nn.Module):
    """
    The main autoregressive Language Model architecture.
    Designed to be fully modular, accepting arbitrary block layers and positional encodings.

    Positional encoding note
    ------------------------
    For ALiBi, `positional_encoding` should be nn.Identity() because ALiBi
    does not modify the embeddings — it is applied inside each attention block
    directly to the attention scores.  The Transformer shell never needs to
    know which positional scheme is in use; that detail lives in the blocks.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        blocks: list,
        positional_encoding: nn.Module,
        dropout: float = 0.1,
    ):
        """
        Args:
            vocab_size (int): Size of the tokenizer's vocabulary (e.g., 50257).
            d_model (int): The hidden embedding dimension.
            blocks (list of nn.Module): A pre-instantiated list of TransformerBlocks.
            positional_encoding (nn.Module): The instantiated positional encoding module.
                                             Pass nn.Identity() for ALiBi.
            dropout (float): Dropout probability for embeddings.
        """
        super().__init__()

        # 1. Token Embedding Matrix
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 2. Positional Encoding (Injected)
        #    nn.Identity() is a safe no-op for ALiBi — x = Identity(x) = x
        self.positional_encoding = positional_encoding
        self.dropout = nn.Dropout(dropout)

        # 3. The Core Network
        self.blocks = nn.ModuleList(blocks)

        # 4. Final LayerNorm (Crucial for Pre-LN architectures)
        self.final_ln = nn.LayerNorm(d_model)

        # 5. Language Modeling (LM) Head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight Tying (GPT-2 standard): shares the embedding and unembedding matrices
        self.token_embedding.weight = self.lm_head.weight

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input token IDs of shape (batch_size, seq_len).
            mask (torch.Tensor, optional): The autoregressive causal mask.
            **kwargs: Extra arguments forwarded to each block (e.g. ALiBi bias).
        Returns:
            torch.Tensor: Logits of shape (batch_size, seq_len, vocab_size).
        """
        # Step 1: Map integer tokens to dense continuous vectors
        x = self.token_embedding(x)

        # Step 2: Inject positional awareness
        #   • Sinusoidal / RoPE  → adds/rotates the embeddings
        #   • ALiBi              → nn.Identity(), so x passes through unchanged
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Step 3: Route through the stack of Transformer blocks
        #   **kwargs carries the ALiBi bias (or nothing for other schemes)
        for block in self.blocks:
            x = block(x, mask=mask, **kwargs)

        # Step 4: Final Layer Normalization
        x = self.final_ln(x)

        # Step 5: Project back to vocabulary dimensions for prediction
        logits = self.lm_head(x)

        return logits