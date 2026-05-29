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

Fix: torch.cos(torch.pi * progress) → math.cos(math.pi * progress)
     torch.pi is a float, but torch.cos expects a tensor; this caused a
     TypeError at runtime. math.cos(math.pi * float) is the correct call.
"""

import sys
import os
import math  # ← used for lr_lambda cosine schedule (not torch.cos/torch.pi)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import wandb

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
from core_ml.models.positional.Sinusoidal   import SinusoidalPositionalEncoding
from core_ml.models.positional.Rope         import RotaryPositionalEmbedding
from core_ml.models.positional.Alibi        import ALiBiPositionalBias
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
    d_model   = cfg.model.d_model
    n_heads   = cfg.model.n_heads
    n_layers  = cfg.model.n_layers
    dropout   = cfg.attention.dropout
    is_causal = cfg.attention.is_causal
    name      = cfg.attention.name

    if name == "vanilla_mha":
        return MultiHeadAttention(d_model, n_heads, n_layers, dropout, is_causal)

    elif name == "sliding_window":
        return SlidingWindowAttention(
            d_model, n_heads,
            n_layers=n_layers,
            window_size=cfg.attention.get("window_size", 256),
            dropout=dropout,
            is_causal=is_causal,
        )

    elif name == "gqa":
        return GroupedQueryAttention(
            d_model, n_heads,
            num_kv_heads=cfg.attention.get("num_kv_heads", 2),
            n_layers=n_layers,
            dropout=dropout,
            is_causal=is_causal,
        )

    elif name == "relu_attention":
        return ReLUAttention(
            d_model, n_heads,
            n_layers=n_layers,
            dropout=dropout,
            is_causal=is_causal,
        )

    elif name == "sparse_attention":
        return SparseAttention(
            d_model, n_heads,
            n_layers=n_layers,
            local_window=cfg.attention.get("local_window", 64),
            stride=cfg.attention.get("stride", 64),
            dropout=dropout,
            is_causal=is_causal,
        )

    else:
        raise ValueError(
            f"Unknown attention type: '{name}'. "
            "Valid: vanilla_mha, sliding_window, gqa, relu_attention, sparse_attention"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: DictConfig) -> nn.Module:
    d_model  = cfg.model.d_model
    n_heads  = cfg.model.n_heads
    n_layers = cfg.model.n_layers
    d_ff     = cfg.model.d_ff
    dropout  = cfg.model.dropout
    pos_name = cfg.positional.name

    # ── 1. Positional encoding setup ─────────────────────────────────────────
    shell_pos   = nn.Identity()
    block_alibi = None
    block_rope  = None

    if pos_name == "sinusoidal":
        shell_pos = SinusoidalPositionalEncoding(
            d_model,
            max_len=cfg.model.max_seq_len,
            base=cfg.positional.get("base", 10000.0),
        )

    elif pos_name == "rope":
        d_head = d_model // n_heads
        block_rope = RotaryPositionalEmbedding(
            d_head,
            max_seq_len=cfg.model.max_seq_len,
            base=cfg.positional.get("base", 10000.0),
        )

    elif pos_name == "alibi":
        block_alibi = ALiBiPositionalBias(n_heads)

    elif pos_name == "relative":
        block_alibi = RelativePositionalBias(
            num_heads=n_heads,
            max_relative_distance=cfg.positional.get("max_relative_distance", 128),
        )

    else:
        raise ValueError(
            f"Unknown positional type: '{pos_name}'. "
            "Valid: sinusoidal, rope, alibi, relative"
        )

    # ── 2. Hybrid config ──────────────────────────────────────────────────────
    is_hybrid   = (cfg.model.name == "hybrid_transformer")
    hybrid_cfg  = cfg.model.get("hybrid", {})
    hybrid_type = hybrid_cfg.get("type", "conv_before_attn") if is_hybrid else None
    kernel_size = hybrid_cfg.get("conv_kernel_size", 3)      if is_hybrid else 3

    # ── 3. Build blocks ───────────────────────────────────────────────────────
    blocks = []
    for layer_idx in range(n_layers):
        attn = _build_attention(cfg)
        ffn  = FeedForward(d_model, d_ff, n_layers, dropout)

        if not is_hybrid:
            block = TransformerBlock(
                d_model, attn, ffn, dropout,
                alibi=block_alibi,
                rope=block_rope,
            )

        elif hybrid_type == "conv_before_attn":
            block = ConvBeforeAttnBlock(
                d_model, attn, ffn, dropout,
                kernel_size=kernel_size,
                alibi=block_alibi,
                rope=block_rope,
            )

        elif hybrid_type == "gated_conv_ffn":
            block = GatedConvFFNBlock(
                d_model, attn, d_ff, dropout,
                kernel_size=kernel_size,
                alibi=block_alibi,
                rope=block_rope,
            )

        elif hybrid_type == "interleaved":
            if layer_idx % 2 == 0:
                block = PureConvBlock(d_model, d_ff, kernel_size, dropout)
            else:
                block = TransformerBlock(
                    d_model, attn, ffn, dropout,
                    alibi=block_alibi,
                    rope=block_rope,
                )

        else:
            raise ValueError(
                f"Unknown hybrid type: '{hybrid_type}'. "
                "Valid: conv_before_attn, gated_conv_ffn, interleaved"
            )

        blocks.append(block)

    # ── 4. Assemble the Transformer shell ─────────────────────────────────────
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
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../..")
    sys.path.insert(0, os.getcwd())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Attention: {cfg.attention.name}  |  Positional: {cfg.positional.name}  |  Model: {cfg.model.name}")

    # ── W&B ──────────────────────────────────────────────────────────────────
    run_name = f"{cfg.experiment_name}_{cfg.attention.name}_{cfg.positional.name}"
    wandb.init(
        project="SAiDL-Core-ML",
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=[cfg.attention.name, cfg.positional.name, cfg.model.name, "training"],
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = prepare_dataloaders(cfg)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.2f} M")
    wandb.config.update({"n_params_M": round(n_params, 2)})
    wandb.watch(model, log="gradients", log_freq=200)

    # ── Throughput Optimizations ──────────────────────────────────────────────
    if device.type == "cuda" and hasattr(torch, "compile"):
        print("Optimizing execution graphs via torch.compile()...")
        model = torch.compile(model)

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        betas=tuple(cfg.training.betas),
        fused=(device.type == "cuda"),  # High-throughput unified memory access kernel
    )
    total_steps = cfg.training.num_epochs * max(
        1, len(train_loader) // cfg.training.gradient_accumulation_steps
    )
    warmup_steps = cfg.training.warmup_iters
    base_lr = cfg.training.learning_rate
    min_lr  = cfg.training.min_lr

    # FIX: use math.cos / math.pi — torch.cos requires a Tensor, not a float.
    # torch.pi is just a float constant so torch.cos(torch.pi * float) raises
    # "RuntimeError: expected Tensor". math.cos takes a plain Python float.
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))   # ← FIXED
        return max(min_lr / base_lr, cosine)

    scheduler = LambdaLR(optimizer, lr_lambda)

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler, cfg, device)
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()