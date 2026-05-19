import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import os
import glob
from tqdm import tqdm

from mechanistic_interpretability.models.sae import TopKSparseAutoencoder

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def train_sae(cfg: DictConfig):
    device = torch.device(cfg.pipeline.device if torch.cuda.is_available() else "cpu")
    print(f"Starting SAE Training on {device}...")

    # 1. Initialize Weights & Biases
    wandb.init(
        project="SAiDL-Interpretability",
        name=f"SAE_{cfg.experiment_name}_k{cfg.sae.k}",
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    # 2. Instantiate the Sparse Autoencoder
    sae = TopKSparseAutoencoder(
        d_model=cfg.pipeline.d_model,
        expansion_factor=cfg.sae.expansion_factor,
        k=cfg.sae.k
    ).to(device)

    # 3. Setup Optimizer
    # Note: We do not use weight decay on the SAE parameters
    optimizer = optim.Adam(sae.parameters(), lr=cfg.training.learning_rate)

    # 4. Find all extracted activation files
    act_files = sorted(glob.glob(os.path.join(cfg.data.save_samples_dir, "*.pt")))
    if not act_files:
        raise FileNotFoundError("No activation files found. Run extract.py first.")

    print(f"Found {len(act_files)} activation files. Beginning training loop.")

    global_step = 0
    sae.train()

    # 5. The Training Loop (Iterating over cached files)
    for epoch in range(cfg.training.num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{cfg.training.num_epochs} ---")
        
        for file_path in act_files:
            # Load a chunk of activations into RAM
            activations = torch.load(file_path, map_location="cpu")
            
            # Create a DataLoader for this specific chunk
            dataset = TensorDataset(activations)
            dataloader = DataLoader(dataset, batch_size=cfg.pipeline.batch_size, shuffle=True)
            
            progress_bar = tqdm(dataloader, desc=f"Training on {os.path.basename(file_path)}")
            
            for batch in progress_bar:
                # x shape: (batch_size, d_model)
                x = batch[0].to(device)
                
                # Forward Pass
                x_reconstructed, feature_acts, l2_loss = sae(x)
                
                # Top-k inherently solves the sparsity constraint, 
                # so our total loss is purely the reconstruction error.
                loss = l2_loss
                
                # Backward Pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # CRITICAL: Re-normalize the decoder weights after every step
                sae.set_decoder_norm_to_unit_norm()
                
                # Calculate metrics for logging
                # L0 norm measures how many neurons are active (should be exactly k)
                l0_norm = (feature_acts > 0).float().sum(dim=-1).mean()
                
                # Calculate variance explained (How much of the original signal did we capture?)
                variance_explained = 1.0 - (l2_loss / x.var(dim=0).mean())
                
                global_step += 1
                
                if global_step % cfg.training.log_interval == 0:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/l0_norm": l0_norm.item(),
                        "train/variance_explained": variance_explained.item(),
                        "global_step": global_step
                    })
                    
                    progress_bar.set_postfix({
                        "Loss": f"{loss.item():.4f}", 
                        "VarExp": f"{variance_explained.item():.2f}"
                    })

    # 6. Save the Final Model
    os.makedirs(cfg.output_dir, exist_ok=True)
    save_path = os.path.join(cfg.output_dir, "sae_final.pt")
    torch.save(sae.state_dict(), save_path)
    print(f"\nTraining Complete. SAE saved to {save_path}")
    
    wandb.finish()

if __name__ == "__main__":
    train_sae()