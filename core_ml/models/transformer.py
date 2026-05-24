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
        super().__init__()

        # 1. Token Embedding Matrix
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 2. Positional Encoding (Injected)
        self.positional_encoding = positional_encoding
        self.dropout = nn.Dropout(dropout)

        # 3. The Core Network
        # (Note: Since blocks are pre-instantiated, the n_layers parameter must 
        # be passed into them in the script where this list is created!)
        self.blocks = nn.ModuleList(blocks)

        # 4. Final LayerNorm (Crucial for Pre-LN architectures)
        self.final_ln = nn.LayerNorm(d_model)

        # 5. Language Modeling (LM) Head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight Tying (GPT-2 standard): shares the embedding and unembedding matrices
        self.token_embedding.weight = self.lm_head.weight

        # CRITICAL FIX: GPT-2 scaled initialization for embeddings.
        # PyTorch defaults to N(0, 1) for embeddings, which causes a massive initial loss spike.
        # Since weights are tied, this single line initializes BOTH the embedding and lm_head.
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

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
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Step 3: Route through the stack of Transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask, **kwargs)

        # Step 4: Final Layer Normalization
        x = self.final_ln(x)

        # Step 5: Project back to vocabulary dimensions for prediction
        logits = self.lm_head(x)

        return logits