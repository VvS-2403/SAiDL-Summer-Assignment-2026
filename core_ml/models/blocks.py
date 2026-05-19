import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    """
    A single standard Transformer block utilizing Pre-Layer Normalization.
    Designed for dependency injection to easily swap Attention and FFN variants.
    """
    def __init__(self, d_model, attention_module, ffn_module, dropout=0.1):
        """
        Args:
            d_model (int): The embedding dimension.
            attention_module (nn.Module): The instantiated attention class (e.g., Vanilla, MQA).
            ffn_module (nn.Module): The instantiated Feed-Forward class.
            dropout (float): Dropout probability.
        """
        super().__init__()
        
        # LayerNorms applied BEFORE the sub-layers (Pre-LN architecture)
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = attention_module 
        
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn = ffn_module
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, **kwargs):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor, optional): Boolean or additive mask for causal/padding attention.
            **kwargs: Extra arguments caught and passed to specialized attention mechanisms (like ALiBi biases).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Block 1: Multi-Head Attention Sub-layer
        # Path: Normalize -> Attend -> Dropout -> Residual Add
        # kwargs is used for alibi
        normalized_x = self.ln_1(x)
        attn_out = self.attn(normalized_x, mask=mask, **kwargs)
        x = x + self.dropout(attn_out)
        
        # Block 2: Feed-Forward Network Sub-layer
        # Path: Normalize -> FFN -> Dropout -> Residual Add
        normalized_x2 = self.ln_2(x)
        ffn_out = self.ffn(normalized_x2)
        x = x + self.dropout(ffn_out)
        
        return x