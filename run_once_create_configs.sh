#!/usr/bin/env bash
# run_once_create_configs.sh
# Creates every missing YAML config that the Hydra setup needs.
# Safe to re-run — will not overwrite files that already exist.
set -e

mkdir -p core_ml/configs/attention
mkdir -p core_ml/configs/positional
mkdir -p core_ml/configs/model

# ── Attention configs ──────────────────────────────────────────────────────────

cat > core_ml/configs/attention/gqa.yaml <<'EOF'
name: "gqa"
num_heads: 8
num_kv_heads: 2
dropout: 0.1
is_causal: true
EOF

cat > core_ml/configs/attention/sliding_window.yaml <<'EOF'
name: "sliding_window"
num_heads: 8
window_size: 256
dropout: 0.1
is_causal: true
EOF

# FIX (BUG 1): was name: "relu" — train.py dispatches on "relu_attention"
cat > core_ml/configs/attention/relu.yaml <<'EOF'
name: "relu_attention"
num_heads: 8
dropout: 0.1
is_causal: true
EOF

# FIX (BUG 2): was name: "sparse" — train.py dispatches on "sparse_attention"
cat > core_ml/configs/attention/sparse.yaml <<'EOF'
name: "sparse_attention"
num_heads: 8
local_window: 64
stride: 64
dropout: 0.1
is_causal: true
EOF

# ── Positional configs ─────────────────────────────────────────────────────────

cat > core_ml/configs/positional/rope.yaml <<'EOF'
name: "rope"
max_len: 4096
base: 10000.0
EOF

cat > core_ml/configs/positional/alibi.yaml <<'EOF'
name: "alibi"
EOF

# FIX (BUG 3): was max_len: 1024 — train.py reads max_relative_distance,
#              RelativePositionalBias.__init__ expects max_relative_distance
cat > core_ml/configs/positional/relative.yaml <<'EOF'
name: "relative"
max_relative_distance: 128
EOF

# ── Model configs ──────────────────────────────────────────────────────────────

# FIX (BUG 4): was flat hybrid_kernel_size key — train.py reads cfg.model.hybrid.type
#              and cfg.model.hybrid.conv_kernel_size from a NESTED hybrid: block
cat > core_ml/configs/model/hybrid.yaml <<'EOF'
name: "hybrid_transformer"
d_model: 512
n_layers: 6
n_heads: 8
d_ff: 2048
dropout: 0.1
vocab_size: 50257
max_seq_len: 1024

hybrid:
  type: "conv_before_attn"   # Options: conv_before_attn | gated_conv_ffn | interleaved
  conv_kernel_size: 3
EOF

# ── hybrid __init__.py ─────────────────────────────────────────────────────────
mkdir -p core_ml/models/hybrid
touch core_ml/models/hybrid/__init__.py

echo "✅ All config files created."
