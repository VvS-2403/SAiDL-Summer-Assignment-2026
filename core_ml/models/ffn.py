import torch
import torch.nn as nn
import math
import wandb

class FeedForward(nn.Module):
    # Change 6: Pass n_layers into the signature
    def __init__(self, d_model, d_ff, n_layers, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        # Change 2: Fix linear_1 weight initialization
        nn.init.normal_(self.linear_1.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.linear_1.bias)

        # Change 1: Fix linear_2 scaled initialization
        # The GPT-2 scale factor prevents variance explosion in the residual stream
        nn.init.normal_(self.linear_2.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))
        nn.init.zeros_(self.linear_2.bias)
        
        # We will use these to temporarily store stats for WandB logging
        self.inter_mean = 0.0
        self.inter_std = 0.0

    def forward(self, x):
        x_inter = self.act(self.linear_1(x))
        
        # Change 4: Check for GELU saturation
        # Storing stats locally rather than calling wandb.log() here to prevent 
        # severe training slowdowns (calling API 6x per forward pass is too slow).
        if self.training:
            with torch.no_grad(): # Don't track gradients for logging metrics
                self.inter_mean = x_inter.mean().item()
                self.inter_std = x_inter.std().item()

        return self.dropout(self.linear_2(x_inter))