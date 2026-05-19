import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard Absolute Sinusoidal Positional Encodings.
    Injects spatial awareness into the model using static sine and cosine waves
    of varying frequencies.
    """
    def __init__(self, d_model: int, max_len: int = 1024, base: float = 10000.0, dropout: float = 0.1):
        """
        Args:
            d_model (int): The embedding dimension.
            max_len (int): The maximum sequence length pre-computed.
            base (float): The geometric progression base for wavelengths.
            dropout (float): Dropout applied after adding the encodings.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 1. Initialize an empty matrix to hold the encodings
        # Shape: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # 2. Create a column vector of position indices [0, 1, 2, ..., max_len - 1]
        # Shape: (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 3. Compute the frequency denominator (div_term)
        # Uses the log-space trick for numerical stability instead of pure division
        # Shape: (d_model / 2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(base) / d_model))
        
        # 4. Apply Sine to even dimensions (0, 2, 4...)
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # 5. Apply Cosine to odd dimensions (1, 3, 5...)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 6. Add a batch dimension for easy broadcasting later
        # Shape: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # 7. Register as a buffer
        # This tells PyTorch: "Save this matrix in the state_dict and move it to the GPU
        # when the model moves, but DO NOT train it with backpropagation."
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Token embeddings of shape (batch_size, seq_len, d_model).
        Returns:
            torch.Tensor: Embeddings with injected positional information.
        """
        seq_len = x.size(1)
        
        # We slice the pre-computed 'pe' tensor up to the current sequence length
        # and mathematically add it to the token embeddings.
        x = x + self.pe[:, :seq_len, :]
        
        return self.dropout(x)