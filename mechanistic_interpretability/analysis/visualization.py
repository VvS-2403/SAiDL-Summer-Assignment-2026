import torch
import hydra
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
import umap
import os
import numpy as np

from mechanistic_interpretability.models.sae import TopKSparseAutoencoder
from mechanistic_interpretability.pipeline.extract import simulate_quantization

def plot_and_save_umap(clean_features: np.ndarray, quantized_features: np.ndarray, save_path: str, bits: int):
    """
    Applies UMAP dimensionality reduction to project high-dimensional SAE features 
    down to 2D space, allowing us to visually inspect quantization damage.
    """
    print("Fitting UMAP manifold... (This may take a minute)")
    
    # Initialize UMAP. We use a fixed random state for reproducible visualizations.
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    
    # Fit the manifold on the clean data to establish the "true" geometric space
    clean_2d = reducer.fit_transform(clean_features)
    
    # Project the quantized data into the EXACT same manifold space
    quantized_2d = reducer.transform(quantized_features)
    
    # Create a side-by-side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Clean Baseline
    ax1.scatter(clean_2d[:, 0], clean_2d[:, 1], alpha=0.5, s=2, c='#1f77b4')
    ax1.set_title("Baseline (FP32) Feature Manifold", fontsize=14)
    ax1.axis('off')
    
    # Plot 2: Quantized Overlay
    ax2.scatter(quantized_2d[:, 0], quantized_2d[:, 1], alpha=0.5, s=2, c='#d62728')
    ax2.set_title(f"Quantized ({bits}-bit) Feature Manifold", fontsize=14)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved manifold visualization to {save_path}")

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def visualize_manifold(cfg: DictConfig):
    device = torch.device(cfg.pipeline.device if torch.cuda.is_available() else "cpu")
    print(f"Starting Visualization Pipeline on {device}...")

    # 1. Load Model, Tokenizer, and SAE
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_path)
    if cfg.data.add_bos_token:
        tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.pretrained_path, 
        output_hidden_states=True
    ).to(device)
    model.eval()

    sae_path = os.path.join(cfg.output_dir, "sae_final.pt")
    sae = TopKSparseAutoencoder(
        d_model=cfg.pipeline.d_model,
        expansion_factor=cfg.sae.expansion_factor,
        k=cfg.sae.k
    ).to(device)
    sae.load_state_dict(torch.load(sae_path, map_location=device))
    sae.eval()

    # 2. Extract a small sample for visualization (e.g., 5000 tokens)
    dataset = load_dataset(cfg.data.name, split="train", streaming=True)
    
    clean_feature_list = []
    quant_feature_list = []
    
    print("Extracting sample activations for visualization...")
    with torch.no_grad():
        batch_texts = []
        for sample in dataset:
            batch_texts.append(sample['text'])
            if len(batch_texts) == 16:  # Small batch
                inputs = tokenizer(batch_texts, max_length=128, truncation=True, return_tensors="pt").to(device)
                
                # Get clean hidden states
                clean_acts = model(**inputs).hidden_states[cfg.pipeline.target_layer]
                flat_clean = clean_acts.view(-1, cfg.pipeline.d_model)
                
                # Get quantized hidden states
                quant_acts = simulate_quantization(clean_acts, cfg.quantization)
                flat_quant = quant_acts.view(-1, cfg.pipeline.d_model)
                
                # Pass both through SAE to get the sparse features
                _, clean_features, _ = sae(flat_clean)
                _, quant_features, _ = sae(flat_quant)
                
                clean_feature_list.append(clean_features.cpu().numpy())
                quant_feature_list.append(quant_features.cpu().numpy())
                
                if sum(len(b) for b in clean_feature_list) > 5000:
                    break
                batch_texts = []

    # 3. Concatenate and Plot
    all_clean = np.concatenate(clean_feature_list, axis=0)
    all_quant = np.concatenate(quant_feature_list, axis=0)
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    save_path = os.path.join(cfg.output_dir, f"umap_projection_{cfg.quantization.bits}bit.png")
    
    plot_and_save_umap(all_clean, all_quant, save_path, cfg.quantization.bits)

if __name__ == "__main__":
    visualize_manifold()