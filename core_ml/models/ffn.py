import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """
    A standard Position-wise Feed-Forward Network (FFN) with a GELU activation.
    Acts as a massive key-value memory bank for the Transformer.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model (int): The embedding dimension (e.g., 512).
            d_ff (int): The hidden expansion dimension, usually 4x d_model (e.g., 2048).
            dropout (float): Dropout probability applied after the activation.
        """
        super().__init__()
        
        # Project up to a higher dimensional space
        self.linear_1 = nn.Linear(d_model, d_ff)
        
        # Gaussian Error Linear Unit (standard for modern LLMs like GPT-2)
        self.activation = nn.GELU()
        
        self.dropout = nn.Dropout(dropout)
        
        # Project back down to the residual stream dimension
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # x shape: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff)
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # x shape: (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        x = self.linear_2(x)
        
        return x