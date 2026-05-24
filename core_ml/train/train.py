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

# Attention variants
from core_ml.models.attention.vanilla_attention import Mul
from core_ml.models.attention.sliding_window import SlidingWindowAttention
from core_ml.models.attention.gqa import GroupedQueryAttention

# Positional variants — filenames are capitalised, imports must match exactly
from core_ml.models.positional.Sinusoidal import SinusoidalPositionalEncoding
from core_ml.models.positional.Rope import RotaryPositionalEmbedding
from core_ml.models.positional.Alibi import ALiBiPositionalBias


def build_model(cfg: DictConfig) -> nn.Module:
    """
    Factory function that dynamically builds the Transformer architecture
    based entirely on the Hydra configuration.

    ALiBi note
    ----------
    ALiBi is NOT an embedding-level positional encoding. It adds a
    distance-proportional bias directly to the attention scores inside
    every block.  Therefore:
      • positional_encoding for the Transformer shell → nn.Identity()
      • An ALiBiPositionalBias instance is created here and injected into
        every attention block so it can be called inside forward() via
        the **kwargs pathway.
    """
    d_model   = cfg.model.d_model
    num_heads = cfg.model.n_heads
    n_layers  = cfg.model.n_layers  # Extract n_layers from config
    use_alibi = cfg.positional.name == "alibi"

    # ------------------------------------------------------------------
    # 1. Select Positional Encoding for the embedding level
    # ------------------------------------------------------------------
    if cfg.positional.name == "sinusoidal":
        pos_module = SinusoidalPositionalEncoding(d_model, cfg.model.max_seq_len)
    elif cfg.positional.name == "rope":
        pos_module = RotaryPositionalEmbedding(d_model, cfg.model.max_seq_len)
    elif cfg.positional.name == "alibi":
        # ALiBi is applied inside attention blocks, not to embeddings.
        # Pass nn.Identity() so Transformer.forward() is a no-op here.
        pos_module = nn.Identity()
    else:
        pos_module = nn.Identity()

    # ------------------------------------------------------------------
    # 2. Optionally build a shared ALiBi bias module
    #    (shared across all blocks; it holds no trainable parameters)
    # ------------------------------------------------------------------
    alibi_module = ALiBiPositionalBias(num_heads) if use_alibi else None

    # ------------------------------------------------------------------
    # 3. Build Transformer Blocks
    # ------------------------------------------------------------------
    blocks = []
    for _ in range(n_layers):
        # Select Attention Mechanism & Pass n_layers for Scaled Initialization
        if cfg.attention.name == "vanilla_mha":
            attn = MultiHeadAttention(
                d_model=d_model, 
                n_heads=num_heads,
                n_layers=n_layers,  # Required for out_proj scaled init
                dropout=cfg.attention.dropout, 
                is_causal=cfg.attention.is_causal
            )
        elif cfg.attention.name == "sliding_window":
            attn = SlidingWindowAttention(
                d_model=d_model, 
                n_heads=num_heads, 
                n_layers=n_layers,  # Required for out_proj scaled init
                window_size=256
            )
        elif cfg.attention.name == "gqa":
            attn = GroupedQueryAttention(
                d_model=d_model, 
                n_heads=num_heads, 
                n_layers=n_layers,  # Required for out_proj scaled init
                num_kv_heads=2
            )
        else:
            raise ValueError(f"Unknown attention type: {cfg.attention.name}")

        # Pass n_layers into the FeedForward for linear_2 scaled init
        ffn = FeedForward(
            d_model=d_model, 
            d_ff=cfg.model.d_ff, 
            n_layers=n_layers,      # Required for linear_2 scaled init
            dropout=cfg.model.dropout
        )

        # Inject the alibi module into each block so it can use it in forward()
        blocks.append(
            TransformerBlock(
                d_model, attn, ffn, cfg.model.dropout,
                alibi=alibi_module      # None for non-ALiBi runs → ignored
            )
        )

    # ------------------------------------------------------------------
    # 4. Assemble full Transformer
    # ------------------------------------------------------------------
    model = Transformer(
        vocab_size=cfg.model.vocab_size,
        d_model=d_model,
        blocks=blocks,
        positional_encoding=pos_module,
        dropout=cfg.model.dropout,
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
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=[cfg.attention.name, cfg.positional.name],
        notes="Baseline sweep over attention variants",
    )

    # 3. Prepare Data
    train_loader, val_loader = prepare_dataloaders(cfg)

    # 4. Build Model
    model = build_model(cfg).to(device)
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    # Watch gradients in W&B (every 200 steps to keep overhead low)
    wandb.watch(model, log="gradients", log_freq=200)

    # 5. Initialize Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        betas=tuple(cfg.training.betas),
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.num_epochs * len(train_loader),
        eta_min=cfg.training.min_lr,
    )

    # 6. Execute Training
    trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler, cfg, device)
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()