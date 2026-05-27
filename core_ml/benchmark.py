"""
core_ml/benchmark.py

Standalone inference benchmark.  For every (attention, positional) combination
that has a saved checkpoint it measures:

  • Validation perplexity at seq_len ∈ {512, 1024, 2048}
  • Inference latency  (ms / batch  and  tokens / sec)
  • Peak GPU memory    (MB)

All results are logged to a dedicated W&B run and also written to a local
JSON file for the report table.

Run from repo root:
    python core_ml/benchmark.py

Hydra overrides are supported as usual:
    python core_ml/benchmark.py attention=gqa positional=rope
"""

import sys
import os

# FIX (BUG 10): benchmark.py lives at core_ml/benchmark.py
# dirname(__file__) = <repo>/core_ml/
# We need ONE level up ("..")  to reach the repo root, not TWO ("../..").
# The old code used "../.." which resolved to the PARENT of the repo.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import math
import time
import glob

import hydra
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from core_ml.data.dataset import LanguageModelingDataset
from core_ml.train.train import build_model


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


def _build_val_loader(seq_len: int, batch_size: int, num_workers: int = 2) -> DataLoader:
    """Builds a fresh WikiText-2 validation loader for a given seq_len."""
    raw = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    tokenized = raw.map(
        lambda ex: tokenizer(ex["text"], truncation=False, return_attention_mask=False),
        batched=True,
        remove_columns=["text"],
    )
    val_stream = [t for sub in tokenized["validation"]["input_ids"] for t in sub]
    val_ds = LanguageModelingDataset(val_stream, seq_len=seq_len)
    return DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True, drop_last=True)


@torch.no_grad()
def _eval_perplexity(model: nn.Module, loader: DataLoader, device: torch.device,
                     use_fp16: bool = True) -> tuple[float, float]:
    """
    Returns (perplexity, tokens_per_sec) over the full loader.
    """
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    t0 = time.time()

    for x, y in tqdm(loader, desc="  eval", leave=False):
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", enabled=use_fp16):
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()

    elapsed = time.time() - t0
    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20))
    tok_per_sec = total_tokens / elapsed
    return ppl, tok_per_sec


@torch.no_grad()
def _latency_benchmark(model: nn.Module, seq_len: int, batch_size: int,
                        device: torch.device, n_warmup: int = 5,
                        n_measure: int = 20) -> dict:
    """
    Measures median inference latency over `n_measure` batches.
    Returns dict with keys: latency_ms, tokens_per_sec, peak_gpu_mb.
    """
    model.eval()
    dummy = torch.randint(0, 50257, (batch_size, seq_len), device=device)

    for _ in range(n_warmup):
        _ = model(dummy)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    _reset_peak_memory()
    times = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        _ = model(dummy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    median_s = sorted(times)[len(times) // 2]
    tokens_per_sec = (batch_size * seq_len) / median_s
    return {
        "latency_ms": round(median_s * 1000, 2),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "peak_gpu_mb": round(_peak_memory_mb(), 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

# config_path is relative to THIS file's location.
# benchmark.py is at core_ml/benchmark.py, so "../configs" → root/configs — correct.
@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Re-anchor CWD and sys.path after Hydra's chdir
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"bench_{cfg.attention.name}_{cfg.positional.name}"
    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {run_name}")
    print(f"{'='*60}")

    wandb.init(
        project="SAiDL-Core-ML",
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=["benchmark", cfg.attention.name, cfg.positional.name],
    )

    # ── 1. Build model ────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    # Try to load best checkpoint if it exists
    ckpt_candidates = glob.glob(
        os.path.join("outputs", "**", "best_model.pt"), recursive=True
    )
    if ckpt_candidates:
        ckpt_path = sorted(ckpt_candidates)[-1]
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        print(f"  Loaded checkpoint: {ckpt_path}")
    else:
        print("  WARNING: No checkpoint found — benchmarking random-init model.")

    model.eval()
    print(f"  Parameters: {n_params:.2f} M")

    results = {
        "attention": cfg.attention.name,
        "positional": cfg.positional.name,
        "n_params_M": round(n_params, 2),
    }

    # ── 2. Perplexity + throughput at multiple sequence lengths ───────────────
    eval_seq_lens = [512, 1024]
    if cfg.model.max_seq_len >= 2048:
        eval_seq_lens.append(2048)

    for seq_len in eval_seq_lens:
        print(f"\n  → seq_len = {seq_len}")
        try:
            loader = _build_val_loader(seq_len, batch_size=4)
            _reset_peak_memory()
            ppl, tok_s = _eval_perplexity(model, loader, device)
            mem_mb = _peak_memory_mb()
            print(f"     PPL={ppl:.2f}  tok/s={tok_s:.0f}  GPU={mem_mb:.0f} MB")

            results[f"ppl_seq{seq_len}"] = round(ppl, 3)
            results[f"tok_per_sec_seq{seq_len}"] = round(tok_s, 1)
            results[f"peak_gpu_mb_seq{seq_len}"] = round(mem_mb, 1)

            wandb.log({
                f"bench/ppl_seq{seq_len}": ppl,
                f"bench/tok_per_sec_seq{seq_len}": tok_s,
                f"bench/peak_gpu_mb_seq{seq_len}": mem_mb,
            })
        except Exception as e:
            print(f"     SKIPPED (seq_len={seq_len}): {e}")
            results[f"ppl_seq{seq_len}"] = None

    # ── 3. Latency micro-benchmark ────────────────────────────────────────────
    print("\n  → Latency micro-benchmark (seq_len=1024, batch=4) ...")
    latency = _latency_benchmark(model, seq_len=1024, batch_size=4, device=device)
    print(f"     {latency}")
    results.update(latency)
    wandb.log({f"bench/{k}": v for k, v in latency.items()})

    # ── 4. Positional extrapolation test ─────────────────────────────────────
    print("\n  → Positional extrapolation probe ...")
    for test_len in [512, 1024, 2048]:
        if test_len > cfg.model.max_seq_len:
            continue
        try:
            loader = _build_val_loader(test_len, batch_size=4)
            ppl, _ = _eval_perplexity(model, loader, device)
            print(f"     extrap ppl @ {test_len}: {ppl:.2f}")
            results[f"extrap_ppl_{test_len}"] = round(ppl, 3)
            wandb.log({f"bench/extrap_ppl_{test_len}": ppl})
        except Exception as e:
            print(f"     SKIPPED extrap {test_len}: {e}")

    # ── 5. Save JSON summary ──────────────────────────────────────────────────
    os.makedirs("benchmark_results", exist_ok=True)
    out_path = f"benchmark_results/{run_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n  Saved: {out_path}")

    wandb.log({"benchmark_summary": wandb.Table(
        columns=list(results.keys()),
        data=[list(results.values())],
    )})

    wandb.finish()
    print(f"\n  DONE — {run_name}")


if __name__ == "__main__":
    main()