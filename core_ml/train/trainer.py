import torch
import torch.nn as nn
import math
import time
import wandb
from tqdm import tqdm
import os

class Trainer:
    """
    Orchestrates the training and evaluation loops for the autoregressive Language Model.
    Implements Automatic Mixed Precision (AMP), gradient accumulation, and WandB logging.
    """
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        
        # Standard loss for Language Modeling
        self.criterion = nn.CrossEntropyLoss()
        
        # Mixed precision scaler for faster/memory-efficient training
        self.scaler = torch.cuda.amp.GradScaler(enabled=(config.training.mixed_precision == "fp16"))
        
        self.global_step = 0
        self.best_val_loss = float('inf')

    def evaluate(self):
        """Runs a full pass over the validation dataset and calculates Perplexity."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for x, y in tqdm(self.val_loader, desc="Evaluating"):
                x, y = x.to(self.device), y.to(self.device)
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=(self.config.training.mixed_precision == "fp16")):
                    logits = self.model(x)
                    
                    # Flatten logits and targets to compute CrossEntropy
                    # logits: (batch * seq_len, vocab_size), y: (batch * seq_len)
                    loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                    
                total_loss += loss.item()
                
        avg_loss = total_loss / len(self.val_loader)
        perplexity = math.exp(avg_loss)
        
        self.model.train()
        return avg_loss, perplexity

    def train(self):
        """Main training loop."""
        self.model.train()
        
        for epoch in range(self.config.training.num_epochs):
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
            
            for step, (x, y) in enumerate(progress_bar):
                x, y = x.to(self.device), y.to(self.device)

                # ── Throughput timer starts here, just before the forward pass ──
                t0 = time.time()

                # 1. Forward Pass with Automatic Mixed Precision
                with torch.cuda.amp.autocast(enabled=(self.config.training.mixed_precision == "fp16")):
                    logits = self.model(x)
                    loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.config.training.gradient_accumulation_steps
                
                # 2. Backward Pass (Scales the gradients to prevent fp16 underflow)
                self.scaler.scale(loss).backward()
                
                # 3. Optimizer Step (executed only after accumulating enough gradients)
                if (step + 1) % self.config.training.gradient_accumulation_steps == 0:
                    # Unscale gradients before clipping so the norm is in real units
                    self.scaler.unscale_(self.optimizer)

                    # Measure grad norm BEFORE clipping so we can log the raw value
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.grad_clip
                    ).item()

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                    
                    self.global_step += 1

                    # ── Throughput calculation ──────────────────────────────────
                    # Tokens processed = one full effective batch
                    # (micro_batch * seq_len * accumulation_steps)
                    step_time = time.time() - t0
                    tokens_processed = (
                        self.config.dataset.batch_size
                        * self.config.dataset.seq_len
                        * self.config.training.gradient_accumulation_steps
                    )
                    tokens_per_sec = tokens_processed / step_time

                    # GPU memory in MB (0 on CPU-only machines)
                    gpu_memory_mb = (
                        torch.cuda.memory_allocated() / 1e6
                        if torch.cuda.is_available() else 0.0
                    )
                    # ───────────────────────────────────────────────────────────

                    # 4. Logging
                    if self.global_step % self.config.training.log_interval == 0:
                        # Unscale the loss back to the true per-step value for logging
                        true_loss = loss.item() * self.config.training.gradient_accumulation_steps

                        wandb.log({
                        "train/loss":           true_loss,
                        "train/perplexity":     math.exp(true_loss),
                        "train/lr":             self.scheduler.get_last_lr()[0],
                        "train/grad_norm":      grad_norm,
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/gpu_memory_mb":  gpu_memory_mb,
                        "epoch":                epoch + 1,
                        "global_step":          self.global_step,
                            }, step=self.global_step)
                        progress_bar.set_postfix({
                            'loss': f"{true_loss:.4f}",
                            'tok/s': f"{tokens_per_sec:.0f}",
                        })
                    
                    # 5. Evaluation
                    if self.global_step % self.config.training.eval_interval == 0:
                        val_loss, val_ppl = self.evaluate()
                        wandb.log({
                            "val/loss":       val_loss,
                            "val/perplexity": val_ppl,
                            "global_step":    self.global_step,
                            # In the log block inside train(), add:
                            "epoch": epoch,
                        })
                        
                        # Save checkpoint if it's the best model so far
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            checkpoint_path = os.path.join(self.config.output_dir, "best_model.pt")
                            torch.save(self.model.state_dict(), checkpoint_path)
                            print(f"\nSaved new best model with Perplexity: {val_ppl:.2f}")
