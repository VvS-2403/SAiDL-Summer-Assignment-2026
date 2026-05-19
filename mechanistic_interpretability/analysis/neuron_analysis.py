import os
import json
import torch
import hydra
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm

from mechanistic_interpretability.models.sae import TopKSparseAutoencoder
from mechanistic_interpretability.pipeline.extract import simulate_quantization

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def analyze_neurons(cfg: DictConfig):
    device = torch.device(cfg.pipeline.device if torch.cuda.is_available() else "cpu")
    print(f"Starting Single-Neuron Analysis Pipeline on {device}...")

    # 1. Load Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_path)
    if cfg.data.add_bos_token:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.pretrained_path, 
        output_hidden_states=True
    ).to(device)
    model.eval()

    # 2. Load the Trained Sparse Autoencoder
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

    # 3. Setup Dataset
    dataset = load_dataset(
        cfg.data.name, 
        split="validation" if "validation" in load_dataset(cfg.data.name, streaming=True).keys() else "train", 
        streaming=True
    )
    
    # We will track the top 10 highest-activating tokens for a subset of features
    # Dictionary structure: { feature_id: [(activation_value, "token_string"), ...] }
    top_activations = defaultdict(list)
    
    # For performance, we only track the first 100 features instead of all 3072
    TRACKED_FEATURES = 100 
    TOKENS_TO_KEEP = 10

    print("Scanning text corpus for maximum activating tokens...")
    with torch.no_grad():
        batch_texts = []
        # We need a decently large sample to find the true max activations
        for sample in tqdm(dataset, desc="Scanning", total=1000):
            batch_texts.append(sample['text'])
            
            if len(batch_texts) == 16:
                inputs = tokenizer(
                    batch_texts, 
                    max_length=128, 
                    truncation=True, 
                    return_tensors="pt"
                ).to(device)
                
                # Forward pass to get clean activations
                clean_acts = model(**inputs).hidden_states[cfg.pipeline.target_layer]
                
                # Apply experimental quantization damage
                test_acts = simulate_quantization(clean_acts, cfg.quantization)
                flat_acts = test_acts.view(-1, cfg.pipeline.d_model)
                
                # Pass through SAE to get feature activations
                _, features, _ = sae(flat_acts)
                
                # Extract token strings
                flat_input_ids = inputs["input_ids"].view(-1)
                tokens = tokenizer.convert_ids_to_tokens(flat_input_ids)
                
                # features shape: (batch * seq_len, d_sae)
                # Iterate over the subset of features we are tracking
                for feature_idx in range(TRACKED_FEATURES):
                    # Get the activation values for this specific feature across all tokens in the batch
                    feature_column = features[:, feature_idx]
                    
                    # Find tokens where this feature actually fired (activation > 0)
                    active_mask = feature_column > 0
                    if not active_mask.any():
                        continue
                        
                    active_vals = feature_column[active_mask].cpu().tolist()
                    active_token_indices = active_mask.nonzero(as_tuple=True)[0].cpu().tolist()
                    
                    # Update our top_activations tracker
                    for val, idx in zip(active_vals, active_token_indices):
                        clean_token = tokens[idx].replace('Ġ', ' ').strip()
                        if clean_token:
                            top_activations[feature_idx].append((val, clean_token))
                            
                    # Sort descending and keep only the top TOKENS_TO_KEEP to save memory
                    top_activations[feature_idx] = sorted(
                        top_activations[feature_idx], 
                        key=lambda x: x[0], 
                        reverse=True
                    )[:TOKENS_TO_KEEP]

                batch_texts = []

    # 4. Format and Save Results
    formatted_results = {}
    for feature_id, activations in top_activations.items():
        # Only save features that actually fired during the scan
        if activations:
            formatted_results[f"Feature_{feature_id}"] = [
                {"token": token, "activation_strength": round(val, 4)} 
                for val, token in activations
            ]

    os.makedirs(cfg.outputs.dir if hasattr(cfg, 'outputs') else cfg.output_dir, exist_ok=True)
    save_path = os.path.join(cfg.outputs.dir if hasattr(cfg, 'outputs') else cfg.output_dir, f"neuron_analysis_{cfg.quantization.bits}bit.json")
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(formatted_results, f, indent=4, ensure_ascii=False)
        
    print(f"\nNeuron Analysis complete. Top activations serialized to {save_path}")

if __name__ == "__main__":
    analyze_neurons()