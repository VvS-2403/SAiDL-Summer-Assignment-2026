import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKSparseAutoencoder(nn.Module):
    """
    A Top-k Sparse Autoencoder (SAE) designed to extract monosemantic features 
    from the dense activation stream of a pre-trained Transformer.
    """
    def __init__(self, d_model: int, expansion_factor: int = 4, k: int = 32):
        """
        Args:
            d_model (int): The residual stream dimension of the target model (e.g., 768 for distilgpt2).
            expansion_factor (int): How much larger the SAE feature space is compared to d_model.
            k (int): The strict number of neurons allowed to fire per token.
        """
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_model * expansion_factor
        self.k = k
        
        # Pre-encoder bias (centers the activations before processing)
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        
        # Encoder: Maps dense residual stream to the sparse, expanded feature space
        self.encoder = nn.Linear(d_model, self.d_sae)
        
        # Decoder: Reconstructs the original residual stream from the sparse features
        self.decoder = nn.Linear(self.d_sae, d_model, bias=False)
        
        # SAEs require the decoder dictionary vectors to have a strict unit norm
        self.set_decoder_norm_to_unit_norm()

    def set_decoder_norm_to_unit_norm(self):
        """
        Normalizes the columns of the decoder matrix. 
        Must be called during initialization and after every optimizer step.
        """
        with torch.no_grad():
            norm = torch.norm(self.decoder.weight, dim=0, keepdim=True)
            # Add small epsilon to prevent division by zero
            self.decoder.weight.data /= (norm + 1e-8)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Extracted activations of shape (batch, seq_len, d_model).
            
        Returns:
            x_reconstructed: The SAE's attempt to recreate 'x'.
            feature_acts: The sparse feature tensor (mostly zeros).
            l2_loss: The Mean Squared Error of the reconstruction.
        """
        # 1. Shift the activations by the geometric center
        x_centered = x - self.b_dec
        
        # 2. Encode to the high-dimensional space
        pre_acts = self.encoder(x_centered)
        
        # 3. Top-k Routing (The Core Sparsity Mechanism)
        # Find the 'k' largest activation values and their specific indices
        topk_vals, topk_indices = torch.topk(pre_acts, self.k, dim=-1)
        
        # Create a blank tensor of exactly zeroes
        feature_acts = torch.zeros_like(pre_acts)
        
        # Scatter the top-k values back into their original index positions.
        # We apply ReLU here to ensure features represent positive geometric directions.
        feature_acts.scatter_(-1, topk_indices, F.relu(topk_vals))
        
        # 4. Reconstruct the original embedding
        x_reconstructed = self.decoder(feature_acts) + self.b_dec
        
        # 5. Calculate Reconstruction Quality
        l2_loss = F.mse_loss(x_reconstructed, x)
        
        return x_reconstructed, feature_acts, l2_loss