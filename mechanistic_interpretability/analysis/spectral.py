import os
import json
import torch
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import numpy as np

from mechanistic_interpretability.models.sae import TopKSparseAutoencoder

def plot_scree(singular_values: np.ndarray, save_path: str, bits: int):
    """
    Generates a Scree Plot to visualize the decay of singular values.
    This shows how the variance/information is distributed across the orthogonal geometric directions.
    """
    print("Generating Spectral Scree Plot...")
    
    # Normalize singular values to show percentage of total variance explained
    explained_variance = (singular_values ** 2) / np.sum(singular_values ** 2)
    cumulative_variance = np.cumsum(explained_variance)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the individual variance explained by each singular vector
    color = 'tab:blue'
    ax1.set_xlabel('Singular Vector Index (Ranked)', fontsize=12)
    ax1.set_ylabel('Variance Explained', color=color, fontsize=12)
    ax1.plot(explained_variance[:500], color=color, alpha=0.8, linewidth=2, label='Individual Variance')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xlim(0, 500) # We zoom in on the first 500 vectors for readability

    # Instantiate a second y-axis that shares the same x-axis for Cumulative Variance
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Cumulative Variance', color=color, fontsize=12)  
    ax2.plot(cumulative_variance[:500], color=color, linestyle='--', linewidth=2, label='Cumulative Variance')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1.05)

    plt.title(f"SVD Scree Plot: SAE Decoder ({bits}-bit Quantization)", fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Scree plot saved to {save_path}")

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def analyze_spectral_properties(cfg: DictConfig):
    device = torch.device(cfg.pipeline.device if torch.cuda.is_available() else "cpu")
    print(f"Starting Spectral (SVD) Analysis Pipeline on {device}...")

    # 1. Load the Trained SAE
    sae_path = os.path.join(cfg.checkpoints.dir if hasattr(cfg, 'checkpoints') else cfg.output_dir, "sae_final_weights.pt")
    if not os.path.exists(sae_path):
        raise FileNotFoundError(f"Trained SAE not found at {sae_path}. Run train_sae.py first.")

    sae = TopKSparseAutoencoder(
        d_model=cfg.pipeline.d_model,
        expansion_factor=cfg.sae.expansion_factor,
        k=cfg.sae.k
    ).to(device)
    sae.load_state_dict(torch.load(sae_path, map_location=device))
    sae.eval()

    # 2. Extract the Decoder Weight Matrix
    # Shape: (d_model, d_sae) e.g., (768, 3072)
    # These are the geometric directions the SAE uses to build concepts.
    decoder_weights = sae.decoder.weight.detach().cpu().float()

    print(f"Computing Singular Value Decomposition on Decoder Matrix of shape {decoder_weights.shape}...")
    
    # 3. Perform SVD
    # U: Left singular vectors, S: Singular values, Vh: Right singular vectors
    # full_matrices=False is computationally much faster for non-square matrices
    U, S, Vh = torch.linalg.svd(decoder_weights, full_matrices=False)
    
    singular_values = S.numpy()
    
    # 4. Calculate Spectral Metrics
    # Effective Dimensionality: How many orthogonal directions do we need to capture 90% of the information?
    explained_var = (singular_values ** 2) / np.sum(singular_values ** 2)
    cumulative_var = np.cumsum(explained_var)
    
    # Find the index where cumulative variance crosses 90%
    eff_dim_90 = int(np.argmax(cumulative_var >= 0.90)) + 1
    
    metrics = {
        "quantization_bits": cfg.quantization.bits,
        "total_singular_values": len(singular_values),
        "max_singular_value": float(singular_values[0]),
        "min_singular_value": float(singular_values[-1]),
        "effective_dimensionality_90pct": eff_dim_90,
        "condition_number": float(singular_values[0] / (singular_values[-1] + 1e-8))
    }
    
    print("\n--- Spectral Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # 5. Save Outputs
    os.makedirs(cfg.outputs.dir if hasattr(cfg, 'outputs') else cfg.output_dir, exist_ok=True)
    
    # Save Metrics JSON
    metrics_path = os.path.join(cfg.outputs.dir if hasattr(cfg, 'outputs') else cfg.output_dir, f"spectral_metrics_{cfg.quantization.bits}bit.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
        
    # Generate and Save Scree Plot
    plot_path = os.path.join(cfg.outputs.dir if hasattr(cfg, 'outputs') else cfg.output_dir, f"spectral_scree_{cfg.quantization.bits}bit.png")
    plot_scree(singular_values, plot_path, cfg.quantization.bits)

if __name__ == "__main__":
    analyze_spectral_properties()