import os
import json
import torch
import hydra
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.cluster import KMeans
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from mechanistic_interpretability.models.sae import TopKSparseAutoencoder
from mechanistic_interpretability.pipeline.extract import simulate_quantization

def perform_semantic_clustering(features: np.ndarray, tokens: list[str], num_clusters: int = 20) -> dict:
    """
    Applies K-Means clustering to the high-dimensional SAE feature vectors,
    grouping tokens with similar semantic representations together.
    """
    print(f"Applying K-Means clustering to discover {num_clusters} semantic concepts...")
    
    # Initialize and fit K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(features)
    
    # Group tokens by their assigned cluster
    clusters = defaultdict(list)
    for token, label in zip(tokens, cluster_labels):
        # Clean up standard GPT-2 byte-pair encoding artifacts (like 'Ġ' for spaces)
        clean_token = token.replace('Ġ', ' ').strip()
        if clean_token:  # Ignore empty tokens
            clusters[int(label)].append(clean_token)
            
    # For each cluster, keep only the unique tokens and limit to the top 15 for readability
    semantic_groups = {}
    for cluster_id, token_list in clusters.items():
        unique_tokens = list(set(token_list))
        # Sort by frequency in the cluster if desired, but here we just take a subset
        semantic_groups[f"Cluster_{cluster_id}"] = unique_tokens[:15]
        
    return semantic_groups

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def cluster_features(cfg: DictConfig):
    device = torch.device(cfg.pipeline.device if torch.cuda.is_available() else "cpu")
    print(f"Starting Clustering Analysis Pipeline on {device}...")

    # 1. Load Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_path)
    if cfg.data.add_bos_token:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.pretrained_path, 
        output_hidden_states=True
    ).to(device)
    model.eval()

    # 2. Load the Trained SAE
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

    # 3. Extract a sample dataset
    dataset = load_dataset(
        cfg.data.name, 
        split="validation" if "validation" in load_dataset(cfg.data.name, streaming=True).keys() else "train", 
        streaming=True
    )
    
    feature_list = []
    token_string_list = []
    
    print("Extracting activations and decoding token strings...")
    with torch.no_grad():
        batch_texts = []
        for sample in tqdm(dataset, desc="Sampling", total=300):
            batch_texts.append(sample['text'])
            if len(batch_texts) == 8:  # Small batch to process memory safely
                inputs = tokenizer(
                    batch_texts, 
                    max_length=128, 
                    truncation=True, 
                    return_tensors="pt"
                ).to(device)
                
                # Get hidden states
                clean_acts = model(**inputs).hidden_states[cfg.pipeline.target_layer]
                
                # Apply experimental condition (baseline vs quantized)
                test_acts = simulate_quantization(clean_acts, cfg.quantization)
                flat_acts = test_acts.view(-1, cfg.pipeline.d_model)
                
                # Pass through the SAE
                _, features, _ = sae(flat_acts)
                
                # We need the actual English text corresponding to every single vector
                flat_input_ids = inputs["input_ids"].view(-1)
                tokens = tokenizer.convert_ids_to_tokens(flat_input_ids)
                
                feature_list.append(features.cpu().numpy())
                token_string_list.extend(tokens)
                
                if len(token_string_list) > 15000:
                    break
                batch_texts = []

    # 4. Perform Clustering
    all_features = np.concatenate(feature_list, axis=0)
    
    # We only cluster tokens that actually activated the SAE (ignore padding/pure zero vectors)
    # A vector is "active" if its sum is greater than 0
    active_indices = np.sum(all_features, axis=1) > 0
    active_features = all_features[active_indices]
    active_tokens = [t for i, t in enumerate(token_string_list) if active_indices[i]]
    
    # Generate Semantic Dictionary
    semantic_clusters = perform_semantic_clustering(
        features=active_features, 
        tokens=active_tokens, 
        num_clusters=25  # Adjust this to find finer or broader concepts
    )
    
    # 5. Save Results
    os.makedirs(cfg.outputs.dir if hasattr(cfg, 'outputs') else cfg.output_dir, exist_ok=True)
    save_path = os.path.join(cfg.outputs.dir if hasattr(cfg, 'outputs') else cfg.output_dir, f"semantic_clusters_{cfg.quantization.bits}bit.json")
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(semantic_clusters, f, indent=4, ensure_ascii=False)
        
    print(f"\nSemantic clustering complete. Results serialized to {save_path}")

if __name__ == "__main__":
    cluster_features()