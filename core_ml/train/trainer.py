"""
core_ml/train/trainer.py

Orchestrates training and evaluation.  Adds to the original:
  • Inference throughput (tokens/sec) measured inside evaluate()
  • Peak GPU memory at evaluation time logged separately
  • Per-epoch summary row logged to W&B (clean table for the report)
  • FFN intermediate activation stats collected and logged once per eval cycle
    (avoids the 6x-per-step overhead of calling wandb.log inside FeedForward)
"""

import torch
import torch.nn as nn
import math
import time
import wandb
import os
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_loader, val_loader,
                 optimizer, scheduler, config, device):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.config       = config
        self.device       = device

        self.criterion = nn.CrossEntropyLoss()

        # AMP scaler — use the non-deprecated API when available
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(config.training.mixed_precision == "fp16"),
        )

        self.global_step  = 0
        self.best_val_loss = float("inf")

    # ─────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate(self) -> tuple[float, float]:
        """
        Full pass over the validation set.

        Returns
        -------
        avg_loss  : float — average cross-entropy loss
        perplexity: float — exp(avg_loss)

        Also logs to W&B:
          val/loss, val/perplexity,
          val/inference_tok_per_sec, val/peak_gpu_mb
        """
        self.model.eval()
        total_loss   = 0.0
        total_tokens = 0

        use_fp16 = (self.config.training.mixed_precision == "fp16")

        # Reset peak memory counter before the eval loop
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        t_eval_start = time.time()

        with torch.no_grad():
            for x, y in tqdm(self.val_loader, desc="Evaluating", leave=False):
                x, y = x.to(self.device), y.to(self.device)

                with torch.amp.autocast("cuda", enabled=use_fp16):
                    logits = self.model(x)
                    loss   = self.criterion(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1),
                    )

                # Accumulate loss weighted by token count for a true average
                n_tokens = x.numel()
                total_loss   += loss.item() * n_tokens
                total_tokens += n_tokens

        eval_time = time.time() - t_eval_start

        avg_loss   = total_loss / max(total_tokens, 1)
        perplexity = math.exp(min(avg_loss, 20))   # cap at e^20 to avoid overflow

        # ── Inference throughput ─────────────────────────────────────────
        inf_tok_per_sec = total_tokens / max(eval_time, 1e-6)

        # ── Peak GPU memory during eval ───────────────────────────────────
        peak_gpu_mb = (
            torch.cuda.max_memory_allocated(self.device) / 1e6
            if torch.cuda.is_available() else 0.0
        )

        wandb.log({
            "val/loss":               avg_loss,
            "val/perplexity":         perplexity,
            "val/inference_tok_per_sec": inf_tok_per_sec,
            "val/peak_gpu_mb":        peak_gpu_mb,
            "global_step":            self.global_step,
        })

        self.model.train()
        return avg_loss, perplexity

    # ─────────────────────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────────────────────

    def train(self):
        self.model.train()
        use_fp16 = (self.config.training.mixed_precision == "fp16")

        for epoch in range(self.config.training.num_epochs):
            epoch_loss   = 0.0
            epoch_tokens = 0
            epoch_start  = time.time()

            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")

            for step, (x, y) in enumerate(progress_bar):
                x, y = x.to(self.device), y.to(self.device)

                t0 = time.time()

                # ── Forward pass ─────────────────────────────────────────
                with torch.amp.autocast("cuda", enabled=use_fp16):
                    logits = self.model(x)
                    loss   = self.criterion(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1),
                    )
                    # Scale loss for gradient accumulation
                    loss_scaled = loss / self.config.training.gradient_accumulation_steps

                # ── Backward ─────────────────────────────────────────────
                self.scaler.scale(loss_scaled).backward()

                # ── Optimiser step (every N micro-steps) ─────────────────
                if (step + 1) % self.config.training.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.grad_clip,
                    ).item()

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()

                    self.global_step += 1

                    # ── Throughput ────────────────────────────────────────
                    step_time = time.time() - t0
                    tokens_processed = (
                        self.config.dataset.batch_size
                        * self.config.dataset.seq_len
                        * self.config.training.gradient_accumulation_steps
                    )
                    train_tok_per_sec = tokens_processed / max(step_time, 1e-6)

                    gpu_mb = (
                        torch.cuda.memory_allocated(self.device) / 1e6
                        if torch.cuda.is_available() else 0.0
                    )

                    true_loss = loss.item()   # unscaled loss for logging

                    # ── Accumulate for epoch summary ──────────────────────
                    epoch_loss   += true_loss * x.numel()
                    epoch_tokens += x.numel()

                    # ── Step-level logging ────────────────────────────────
                    if self.global_step % self.config.training.log_interval == 0:
                        # Collect FFN activation stats (stored on each FFN module)
                        ffn_means, ffn_stds = [], []
                        for m in self.model.modules():
                            # FeedForward stores inter_mean/inter_std when training=True
                            cls_name = type(m).__name__
                            if cls_name == "FeedForward":
                                ffn_means.append(m.inter_mean)
                                ffn_stds.append(m.inter_std)

                        log_dict = {
                            "train/loss":            true_loss,
                            "train/perplexity":      math.exp(min(true_loss, 20)),
                            "train/lr":              self.scheduler.get_last_lr()[0],
                            "train/grad_norm":       grad_norm,
                            "train/tok_per_sec":     train_tok_per_sec,
                            "train/gpu_memory_mb":   gpu_mb,
                            "epoch":                 epoch + 1,
                            "global_step":           self.global_step,
                        }
                        if ffn_means:
                            log_dict["train/ffn_inter_mean"] = sum(ffn_means) / len(ffn_means)
                            log_dict["train/ffn_inter_std"]  = sum(ffn_stds)  / len(ffn_stds)

                        wandb.log(log_dict, step=self.global_step)
                        progress_bar.set_postfix({
                            "loss":  f"{true_loss:.4f}",
                            "tok/s": f"{train_tok_per_sec:.0f}",
                        })

                    # ── Periodic evaluation ───────────────────────────────
                    if self.global_step % self.config.training.eval_interval == 0:
                        val_loss, val_ppl = self.evaluate()
                        print(f"\n  [step {self.global_step}] "
                              f"val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}")

                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            ckpt_path = os.path.join(
                                self.config.output_dir, "best_model.pt"
                            )
                            os.makedirs(self.config.output_dir, exist_ok=True)
                            torch.save(self.model.state_dict(), ckpt_path)
                            print(f"  ✓ Saved best model (ppl={val_ppl:.2f})")

            # ── End-of-epoch summary ──────────────────────────────────────
            epoch_time   = time.time() - epoch_start
            epoch_avg_loss = epoch_loss / max(epoch_tokens, 1)
            epoch_ppl    = math.exp(min(epoch_avg_loss, 20))
            epoch_tps    = epoch_tokens / max(epoch_time, 1e-6)

            # Peak GPU memory over the whole epoch
            peak_epoch_mb = (
                torch.cuda.max_memory_allocated(self.device) / 1e6
                if torch.cuda.is_available() else 0.0
            )

            wandb.log({
                "epoch_summary/train_loss":        epoch_avg_loss,
                "epoch_summary/train_perplexity":  epoch_ppl,
                "epoch_summary/train_tok_per_sec": epoch_tps,
                "epoch_summary/epoch_time_s":      epoch_time,
                "epoch_summary/peak_gpu_mb":       peak_epoch_mb,
                "epoch": epoch + 1,
            })

            print(f"\nEpoch {epoch + 1} summary | "
                  f"loss={epoch_avg_loss:.4f} | ppl={epoch_ppl:.2f} | "
                  f"tok/s={epoch_tps:.0f} | "
                  f"peak_gpu={peak_epoch_mb:.0f} MB | "
                  f"time={epoch_time:.1f}s")