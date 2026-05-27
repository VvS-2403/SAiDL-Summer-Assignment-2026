"""
core_ml/train/trainer.py

Complete training and evaluation loop.

Fixes vs. old version:
  1. torch.cuda.amp -> torch.amp  (non-deprecated API, PyTorch 2.0+)
  2. evaluate() now logs:
       val/loss, val/perplexity,
       val/inference_tok_per_sec,   <- was missing
       val/peak_gpu_mb              <- was missing
  3. train() now logs:
       train/tok_per_sec            <- was missing
       train/gpu_memory_mb          <- was missing
       train/grad_norm
       train/lr
       train/ffn_inter_mean / std
       epoch_summary/* after every epoch
  4. Final evaluation after all epochs complete
  5. Progress bar shows loss + ppl + tok/s
"""

import torch
import torch.nn as nn
import math
import time
import os

import wandb
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        config,
        device: torch.device,
    ):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.config       = config
        self.device       = device

        self.criterion = nn.CrossEntropyLoss()

        # ── AMP: use the non-deprecated torch.amp API ─────────────────────────
        self.use_fp16 = (config.training.mixed_precision == "fp16")
        self.scaler   = torch.amp.GradScaler("cuda", enabled=self.use_fp16)

        self.global_step   = 0
        self.best_val_loss = float("inf")

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _peak_gpu_mb(self) -> float:
        """Returns peak GPU memory allocated since last reset, in MB."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(self.device) / 1e6
        return 0.0

    def _reset_peak_gpu(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def _current_gpu_mb(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(self.device) / 1e6
        return 0.0

    def _collect_ffn_stats(self):
        """
        Collects mean/std of FFN intermediate activations stored on each
        FeedForward module. Called once per log interval — far cheaper than
        calling wandb.log inside every forward pass.
        """
        means, stds = [], []
        for m in self.model.modules():
            if type(m).__name__ == "FeedForward":
                means.append(m.inter_mean)
                stds.append(m.inter_std)
        if means:
            return sum(means) / len(means), sum(stds) / len(stds)
        return None, None

    # ─────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float]:
        """
        Full pass over the validation set.

        Returns
        -------
        avg_loss   : float
        perplexity : float

        Logs to W&B
        -----------
        val/loss
        val/perplexity
        val/inference_tok_per_sec   ← tokens processed per second during inference
        val/peak_gpu_mb             ← peak GPU memory during this eval pass
        """
        self.model.eval()
        total_loss   = 0.0
        total_tokens = 0

        self._reset_peak_gpu()
        t_start = time.perf_counter()

        for x, y in tqdm(self.val_loader, desc="  Eval", leave=False):
            x, y = x.to(self.device), y.to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_fp16):
                logits = self.model(x)
                loss   = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                )

            n_toks        = x.numel()
            total_loss   += loss.item() * n_toks
            total_tokens += n_toks

        elapsed = time.perf_counter() - t_start

        avg_loss   = total_loss / max(total_tokens, 1)
        perplexity = math.exp(min(avg_loss, 20))   # cap at e^20 to avoid overflow

        # ── Inference throughput ──────────────────────────────────────────────
        inf_tok_per_sec = total_tokens / max(elapsed, 1e-9)

        # ── Peak GPU memory during this eval pass ─────────────────────────────
        peak_mb = self._peak_gpu_mb()

        wandb.log(
            {
                "val/loss":                  avg_loss,
                "val/perplexity":            perplexity,
                "val/inference_tok_per_sec": inf_tok_per_sec,
                "val/peak_gpu_mb":           peak_mb,
            },
            step=self.global_step,
        )

        self.model.train()
        return avg_loss, perplexity

    # ─────────────────────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────────────────────

    def train(self):
        cfg       = self.config
        log_every = cfg.training.log_interval
        eval_every = cfg.training.eval_interval
        accum_steps = cfg.training.gradient_accumulation_steps

        self.model.train()

        for epoch in range(cfg.training.num_epochs):
            epoch_loss   = 0.0
            epoch_tokens = 0
            epoch_start  = time.perf_counter()
            self._reset_peak_gpu()

            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{cfg.training.num_epochs}",
            )

            for step, (x, y) in enumerate(pbar):
                x, y = x.to(self.device), y.to(self.device)
                step_start = time.perf_counter()

                # ── Forward pass ──────────────────────────────────────────────
                with torch.amp.autocast("cuda", enabled=self.use_fp16):
                    logits = self.model(x)
                    loss   = self.criterion(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1),
                    )
                    # Scale loss for gradient accumulation
                    scaled_loss = loss / accum_steps

                # ── Backward pass ─────────────────────────────────────────────
                self.scaler.scale(scaled_loss).backward()

                # ── Optimiser step every accum_steps micro-batches ────────────
                if (step + 1) % accum_steps == 0:
                    # Unscale before grad clip so clip norm is on the real grads
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        cfg.training.grad_clip,
                    ).item()

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                    self.global_step += 1

                    # ── Throughput ────────────────────────────────────────────
                    step_secs = time.perf_counter() - step_start
                    # Tokens processed across all micro-batches in this accum window
                    tokens_this_step = (
                        cfg.dataset.batch_size
                        * cfg.dataset.seq_len
                        * accum_steps
                    )
                    train_tok_per_sec = tokens_this_step / max(step_secs, 1e-9)

                    true_loss = loss.item()   # unscaled, for logging
                    ppl       = math.exp(min(true_loss, 20))

                    epoch_loss   += true_loss * x.numel()
                    epoch_tokens += x.numel()

                    # ── Step-level W&B logging ────────────────────────────────
                    if self.global_step % log_every == 0:
                        ffn_mean, ffn_std = self._collect_ffn_stats()
                        log_dict = {
                            "train/loss":          true_loss,
                            "train/perplexity":    ppl,
                            "train/lr":            self.scheduler.get_last_lr()[0],
                            "train/grad_norm":     grad_norm,
                            "train/tok_per_sec":   train_tok_per_sec,
                            "train/gpu_memory_mb": self._current_gpu_mb(),
                            "epoch":               epoch + 1,
                        }
                        if ffn_mean is not None:
                            log_dict["train/ffn_inter_mean"] = ffn_mean
                            log_dict["train/ffn_inter_std"]  = ffn_std

                        wandb.log(log_dict, step=self.global_step)

                    pbar.set_postfix(
                        loss=f"{true_loss:.4f}",
                        ppl=f"{ppl:.1f}",
                        tok_s=f"{train_tok_per_sec:.0f}",
                    )

                    # ── Periodic evaluation ───────────────────────────────────
                    if self.global_step % eval_every == 0:
                        val_loss, val_ppl = self.evaluate()
                        print(
                            f"\n  [step {self.global_step:>6d}] "
                            f"val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}"
                        )
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self._save_checkpoint(val_ppl)

            # ── End-of-epoch summary ──────────────────────────────────────────
            epoch_secs    = time.perf_counter() - epoch_start
            epoch_avg_loss = epoch_loss / max(epoch_tokens, 1)
            epoch_ppl      = math.exp(min(epoch_avg_loss, 20))
            epoch_tps      = epoch_tokens / max(epoch_secs, 1e-9)
            peak_epoch_mb  = self._peak_gpu_mb()

            wandb.log(
                {
                    "epoch_summary/train_loss":        epoch_avg_loss,
                    "epoch_summary/train_perplexity":  epoch_ppl,
                    "epoch_summary/train_tok_per_sec": epoch_tps,
                    "epoch_summary/epoch_time_s":      epoch_secs,
                    "epoch_summary/peak_gpu_mb":       peak_epoch_mb,
                    "epoch": epoch + 1,
                },
                step=self.global_step,
            )

            print(
                f"\nEpoch {epoch + 1} done | "
                f"loss={epoch_avg_loss:.4f} | ppl={epoch_ppl:.2f} | "
                f"tok/s={epoch_tps:.0f} | peak_gpu={peak_epoch_mb:.0f} MB | "
                f"time={epoch_secs:.1f}s"
            )

        # ── Final evaluation after all epochs ─────────────────────────────────
        print("\nRunning final evaluation...")
        final_loss, final_ppl = self.evaluate()
        wandb.log(
            {"final/val_loss": final_loss, "final/val_perplexity": final_ppl},
            step=self.global_step,
        )
        print(f"Final  val_loss={final_loss:.4f}  val_ppl={final_ppl:.2f}")
        return final_loss, final_ppl

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpoint
    # ─────────────────────────────────────────────────────────────────────────

    def _save_checkpoint(self, val_ppl: float):
        out_dir = self.config.output_dir
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "best_model.pt")
        torch.save(self.model.state_dict(), path)
        print(f"  ✓ New best checkpoint saved  val_ppl={val_ppl:.2f}  → {path}")
