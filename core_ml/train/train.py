import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import os

# Internal imports from our repository
from core_ml.train.dataset import prepare_dataloaders
from core_ml.train.trainer import Trainer
from core_ml.models.transformer import Transformer
from core_ml.models.blocks import TransformerBlock
from core_ml.models.ffn import FeedForward

# Attention and Positional variants
from core_ml.models.attention.vanilla_attention import MultiHeadAttention
from core_ml.models.attention.sliding_window import SlidingWindowAttention
from core_ml.models.attention.gqa import GroupedQueryAttention
from core_ml.models.positional.sinusoidal import SinusoidalPositionalEncoding
from core_ml.models.positional.rope import RotaryPositionalEmbedding
from core_ml.models.positional.alibi import ALiBiPositionalBias

def build_model(cfg: DictConfig) -> nn.Module:
    """
    Factory function that dynamically builds the Transformer architecture
    based entirely on the Hydra configuration.
    """
    d_model = cfg.model.d_model
    num_heads = cfg.model.n_heads
    
    # 1. Select Positional Encoding
    if cfg.positional.name == "sinusoidal":
        pos_module = SinusoidalPositionalEncoding(d_model, cfg.model.max_seq_len)
    elif cfg.positional.name == "rope":
        pos_module = RotaryPositionalEmbedding(d_model, cfg.model.max_seq_len)
    elif cfg.positional.name == "alibi":
        pos_module = ALiBiPositionalBias(num_heads)
    else:
        # Fallback to empty Identity if testing un-encoded baselines
        pos_module = nn.Identity()

    # 2. Build Transformer Blocks
    blocks = []
    for _ in range(cfg.model.n_layers):
        # Select Attention Mechanism
        if cfg.attention.name == "vanilla_mha":
            attn = MultiHeadAttention(d_model, num_heads, cfg.attention.dropout, cfg.attention.is_causal)
        elif cfg.attention.name == "sliding_window":
            attn = SlidingWindowAttention(d_model, num_heads, window_size=256)
        elif cfg.attention.name == "gqa":
            attn = GroupedQueryAttention(d_model, num_heads, num_kv_heads=2)
        else:
            raise ValueError(f"Unknown attention type: {cfg.attention.name}")
            
        ffn = FeedForward(d_model, cfg.model.d_ff, cfg.model.dropout)
        blocks.append(TransformerBlock(d_model, attn, ffn, cfg.model.dropout))

    # 3. Assemble full Transformer
    model = Transformer(
        vocab_size=cfg.model.vocab_size,
        d_model=d_model,
        blocks=blocks,
        positional_encoding=pos_module,
        dropout=cfg.model.dropout
    )
    return model

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")
    
    # 2. Initialize Weights & Biases
    wandb.init(
        project="SAiDL-Core-ML",
        name=f"{cfg.experiment_name}_{cfg.attention.name}_{cfg.positional.name}",
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    # 3. Prepare Data
    train_loader, val_loader = prepare_dataloaders(cfg)
    
    # 4. Build Model
    model = build_model(cfg).to(device)
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    
    # 5. Initialize Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=cfg.training.learning_rate, 
        weight_decay=cfg.training.weight_decay,
        betas=tuple(cfg.training.betas)
    )
    
    # Note: In a production run, you would use a Warmup wrapper here. 
    # For baseline simplicity, we use CosineAnnealingLR.
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.num_epochs * len(train_loader), eta_min=cfg.training.min_lr)
    
    # 6. Execute Training
    trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler, cfg, device)
    trainer.train()
    
    wandb.finish()

if __name__ == "__main__":
    main()