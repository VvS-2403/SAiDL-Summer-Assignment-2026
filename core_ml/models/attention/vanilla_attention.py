import torch.nn as nn
import math

class VanillaAttention(nn.Module):
    # Change 6: Add n_layers to signature
    def __init__(self, d_model, n_heads, n_layers, dropout=0.1):
        super().__init__()
        # ... your existing setup ...
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Change 3: Fix Attention initializations
        nn.init.normal_(self.qkv_proj.weight, mean=0.0, std=0.02)
        if self.qkv_proj.bias is not None:
            nn.init.zeros_(self.qkv_proj.bias)

        # Scaled init for output projection
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
            
    # ... rest of your code ...
