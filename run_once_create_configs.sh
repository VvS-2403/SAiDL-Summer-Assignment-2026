#!/usr/bin/env bash
set -e

mkdir -p core_ml/configs/attention
mkdir -p core_ml/configs/positional
mkdir -p core_ml/configs/model

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

cat > core_ml/configs/attention/relu.yaml <<'EOF'
name: "relu"
num_heads: 8
dropout: 0.1
is_causal: true
EOF

cat > core_ml/configs/attention/sparse.yaml <<'EOF'
name: "sparse"
num_heads: 8
local_window: 64
stride: 64
dropout: 0.1
is_causal: true
EOF

cat > core_ml/configs/positional/rope.yaml <<'EOF'
name: "rope"
max_len: 4096
base: 10000.0
EOF

cat > core_ml/configs/positional/alibi.yaml <<'EOF'
name: "alibi"
EOF

cat > core_ml/configs/positional/relative.yaml <<'EOF'
name: "relative"
max_len: 1024
EOF

cat > core_ml/configs/model/hybrid.yaml <<'EOF'
name: "hybrid_transformer"
d_model: 512
n_layers: 6
n_heads: 8
d_ff: 2048
dropout: 0.1
vocab_size: 50257
max_seq_len: 1024
hybrid_kernel_size: 3
EOF

echo "Created missing YAML configuration files."
