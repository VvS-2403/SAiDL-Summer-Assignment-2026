"""
core_ml/train/train.py

Entry point for all Core ML training runs.
Controlled entirely by Hydra configs — swap attention, positional, and model
type from the command line with no code changes.

Supported attention variants:
    vanilla_mha | sliding_window | gqa | relu_attention | sparse_attention

Supported positional variants:
    sinusoidal | rope | alibi | relative

Supported model types (cfg.model.name):
    baseline_transformer | hybrid_transformer

Hybrid sub-types (cfg.model.hybrid.type):
    conv_before_attn | gated_conv_ffn | interleaved
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import os

# ── Internal imports ──────────────────────────────────────────────────────────
from core_ml.train.dataset  import prepare_dataloaders
from core_ml.train.trainer  import Trainer
from core_ml.models.transformer import Transformer
from core_ml.models.blocks      import TransformerBlock
from core_ml.models.ffn         import FeedForward

# Attention variants
from core_ml.models.attention.vanilla_attention import MultiHeadAttention
from core_ml.models.attention.sliding_window    import SlidingWindowAttention
from core_ml.models.attention.gqa               import GroupedQueryAttention
from core_ml.models.attention.relu_attention    import ReLUAttention
from core_ml.models.attention.Sparse_attention  import SparseAttention

# Positional variants
from core_ml.models.positional.Sinusoidal  import SinusoidalPositionalEncoding
from core_ml.models.positional.Rope        import RotaryPositionalEmbedding
from core_ml.models.positional.Alibi       import ALiBiPositionalBias
from core_ml.models.positional.RelativeBias import RelativePositionalBias

# Hybrid blocks
from core_ml.models.hybrid.hybrid_blocks import (
    ConvBeforeAttnBlock,
    GatedConvFFNBlock,
    PureConvBlock,
)


# ─────────────────────────────────────────────────────────────────────────────
# Attention factory
# ─────────────────────────────────────────────────────────────────────────────

def _build_attention(cfg: DictConfig) -> nn.Module:
    """Instantiates the correct attention module from config."""
    d_model   = cfg.model.d_model
    n_heads   = cfg.model.n_heads
    n_layers  = cfg.model.n_layers
    dropout   = cfg.attention.dropout
    is_causal = cfg.attention.is_causal

    name = cfg.attention.name

    if name == "vanilla_mha":
        return MultiHeadAttention(d_model, n_heads, n_layers, dropout, is_causal)

    elif name == "sliding_window":
        return SlidingWindowAttention(
            d_model, n_heads, n_layers=n_layers,
            window_size=cfg.attention.get("window_size", 256),
            dropout=dropout, is_causal=is_causal,
        )

    elif name == "gqa":
        return GroupedQueryAttention(
            d_model, n_heads,
            num_kv_heads=cfg.attention.get("num_kv_heads", 2),
            dropout=dropout, is_causal=is_causal,
        )

    elif name == "relu_attention":
        return ReLUAttention(d_model, n_heads, dropout=dropout, is_causal=is_causal)

    elif name == "sparse_attention":
        return SparseAttention(
            d_model, n_heads,
            local_window=cfg.attention.get("local_window", 64),
            stride=cfg.attention.get("stride", 64),
            dropout=dropout, is_causal=is_causal,
        )

    else:
        raise ValueError(f"Unknown attention type: '{name}'. "
                         "Valid: vanilla_mha, sliding_window, gqa, "
                         "relu_attention, sparse_attention")


# ─────────────────────────────────────────────────────────────────────────────
# Main model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: DictConfig) -> nn.Module:
    """
    Builds the full Transformer from the Hydra config.

    Positional encoding dispatch
    ----------------------------
    sinusoidal : added to token embeddings inside the Transformer shell (classic).
    rope       : the RoPE module is created here and injected into every block;
                 each block passes it into attention.forward() via kwargs so Q and K
                 are rotated AFTER head-splitting.  nn.Identity() is passed as the
                 shell's positional_encoding so the shell does nothing extra.
    alibi      : ALiBiPositionalBias injected into every block; computed on-the-fly
                 and added to attention scores. nn.Identity() for the shell.
    relative   : RelativePositionalBias injected the same way as ALiBi.
                 nn.Identity() for the shell.
    """
    d_model  = cfg.model.d_model
    n_heads  = cfg.model.n_heads
    n_layers = cfg.model.n_layers
    d_ff     = cfg.model.d_ff
    dropout  = cfg.model.dropout
    pos_name = cfg.positional.name

    # ── 1. Positional encoding setup ─────────────────────────────────────────
    # shell_pos  : passed to Transformer.__init__ (operates on flat embeddings)
    # block_alibi: passed to each TransformerBlock as the `alibi` arg
    # block_rope : passed to each TransformerBlock as the `rope` arg

    shell_pos   = nn.Identity()   # default: no-op at embedding level
    block_alibi = None
    block_rope  = None

    if pos_name == "sinusoidal":
        shell_pos = SinusoidalPositionalEncoding(
            d_model,
            max_len=cfg.model.max_seq_len,
            base=cfg.positional.get("base", 10000.0),
        )

    elif pos_name == "rope":
        # RoPE is NOT an embedding-level encoding.
        # The module is shared across all blocks; it holds only cached sin/cos tables.
        block_rope = RotaryPositionalEmbedding(
            d_model,
            max_seq_len=cfg.model.max_seq_len,
            base=cfg.positional.get("base", 10000.0),
        )
        # shell_pos stays nn.Identity()

    elif pos_name == "alibi":
        block_alibi = ALiBiPositionalBias(n_heads)
        # shell_pos stays nn.Identity()

    elif pos_name == "relative":
        block_alibi = RelativePositionalBias(
            num_heads=n_heads,
            max_relative_distance=cfg.positional.get("max_relative_distance", 128),
        )
        # shell_pos stays nn.Identity()

    else:
        raise ValueError(f"Unknown positional type: '{pos_name}'. "
                         "Valid: sinusoidal, rope, alibi, relative")

    # ── 2. Build blocks ───────────────────────────────────────────────────────
    is_hybrid = (cfg.model.name == "hybrid_transformer")
    hybrid_type = cfg.model.get("hybrid", {}).get("type", "conv_before_attn") if is_hybrid else None
    kernel_size = cfg.model.get("hybrid", {}).get("conv_kernel_size", 3) if is_hybrid else 3

    blocks = []
    for layer_idx in range(n_layers):
        attn = _build_attention(cfg)
        ffn  = FeedForward(d_model, d_ff, n_layers, dropout)

        if not is_hybrid:
            # ── Standard TransformerBlock ─────────────────────────────────
            block = TransformerBlock(
                d_model, attn, ffn, dropout,
                alibi=block_alibi,
                rope=block_rope,
            )

        elif hybrid_type == "conv_before_attn":
            # ── Conv1D prepended to every block ───────────────────────────
            block = ConvBeforeAttnBlock(
                d_model, attn, ffn, dropout,
                kernel_size=kernel_size,
                alibi=block_alibi,
                # ConvBeforeAttnBlock uses the same alibi kwarg pattern;
                # RoPE is forwarded through its **kwargs into attention.
            )
            # Attach rope so ConvBeforeAttnBlock forwards it
            block.rope = block_rope

        elif hybrid_type == "gated_conv_ffn":
            # ── Standard attention + gated conv FFN ──────────────────────
            block = GatedConvFFNBlock(
                d_model, attn, d_ff, dropout,
                kernel_size=kernel_size,
                alibi=block_alibi,
            )
            block.rope = block_rope

        elif hybrid_type == "interleaved":
            # ── Alternate: even layers = conv-only, odd = attention ───────
            if layer_idx % 2 == 0:
                block = PureConvBlock(d_model, d_ff, kernel_size, dropout)
            else:
                block = TransformerBlock(
                    d_model, attn, ffn, dropout,
                    alibi=block_alibi,
                    rope=block_rope,
                )
        else:
            raise ValueError(f"Unknown hybrid type: '{hybrid_type}'. "
                             "Valid: conv_before_attn, gated_conv_ffn, interleaved")

        blocks.append(block)

    # ── 3. Assemble Transformer ───────────────────────────────────────────────
    model = Transformer(
        vocab_size=cfg.model.vocab_size,
        d_model=d_model,
        blocks=blocks,
        positional_encoding=shell_pos,
        dropout=dropout,
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Hydra entry point
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── W&B ──────────────────────────────────────────────────────────────────
    run_name = f"{cfg.experiment_name}_{cfg.attention.name}_{cfg.positional.name}"
    wandb.init(
        project="SAiDL-Core-ML",
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=[cfg.attention.name, cfg.positional.name,
              cfg.model.name, "training"],
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = prepare_dataloaders(cfg)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.2f} M")
    wandb.config.update({"n_params_M": round(n_params, 2)})

    # Log gradient histograms every 200 steps
    wandb.watch(model, log="gradients", log_freq=200)

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
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

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler, cfg, device)
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()