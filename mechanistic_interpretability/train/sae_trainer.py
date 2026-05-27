import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import wandb
from tqdm import tqdm

from mechanistic_interpretability.models.sae import TopKSparseAutoencoder


class SAETrainer:
    """Reusable SAE training utility for cached activation datasets."""

    def __init__(self, sae: TopKSparseAutoencoder, cfg, device: torch.device):
        self.sae = sae
        self.cfg = cfg
        self.device = device
        self.optimizer = optim.Adam(
            sae.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=getattr(cfg.training, 'weight_decay', 0.0),
        )
        self.global_step = 0

    def train_chunk(self, activations: torch.Tensor):
        dataset = TensorDataset(activations)
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.pipeline.batch_size,
            shuffle=True,
            drop_last=True,
        )

        self.sae.train()
        for (x,) in tqdm(loader, desc='SAE training batch', leave=False):
            x = x.to(self.device)
            x_hat, feature_acts, l2_loss = self.sae(x)
            loss = l2_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.sae.set_decoder_norm_to_unit_norm()

            self.global_step += 1
            if self.global_step % self.cfg.training.log_interval == 0:
                l0 = (feature_acts > 0).float().sum(dim=-1).mean().item()
                var_exp = float(1.0 - l2_loss.item() / (x.var(dim=0).mean().item() + 1e-8))
                wandb.log(
                    {
                        'sae/loss': loss.item(),
                        'sae/l0_norm': l0,
                        'sae/variance_explained': var_exp,
                        'global_step': self.global_step,
                    }
                )

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.sae.state_dict(), path)
