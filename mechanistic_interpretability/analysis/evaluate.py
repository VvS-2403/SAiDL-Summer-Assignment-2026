import torch
import hydra
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json
import os

from mechanistic_interpretability.models.sae import TopKSparseAutoencoder
from mechanistic_interpretability.pipeline.extract import simulate_quantization

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def evaluate_sae(cfg: DictConfig):
    device = torch.device(cfg.pipeline.device if torch.cuda.is_available() else "cpu")
    print(f"Starting SAE Evaluation on {device}...")
    print(f"Testing Quantization Mode: {cfg.quantization.bits}-bit {cfg.quantization.method}")

    # 1. Load the Target Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_path)
    if cfg.data.add_bos_token:
        tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.pretrained_path, 
        output_hidden_states=cfg.model.output_hidden_states
    ).to(device)
    model.eval()

    # 2. Load the Trained SAE
    sae_path = os.path.join(cfg.output_dir, "sae_final.pt")
    if not os.path.exists(sae_path):
        raise FileNotFoundError(f"Trained SAE not found at {sae_path}. Run train_sae.py first.")
        
    sae = TopKSparseAutoencoder(
        d_model=cfg.pipeline.d_model,
        expansion_factor=cfg.sae.expansion_factor,
        k=cfg.sae.k
    ).to(device)
    sae.load_state_dict(torch.load(sae_path, map_location=device))
    sae.eval()

    # 3. Load Validation Dataset Stream
    dataset = load_dataset(
        cfg.data.name, 
        split="validation" if "validation" in load_dataset(cfg.data.name, streaming=True).keys() else "train", 
        streaming=True,
        trust_remote_code=True
    )

    # 4. Evaluation Metrics
    total_l2_loss = 0.0
    total_variance_explained = 0.0
    total_l0_norm = 0.0
    batches_processed = 0

    print("Running evaluation loop...")
    with torch.no_grad():
        batch_texts = []
        for sample in tqdm(dataset, total=1000): # Evaluate on a smaller subset for speed
            batch_texts.append(sample['text'])
            
            if len(batch_texts) == cfg.pipeline.batch_size:
                inputs = tokenizer(
                    batch_texts, 
                    max_length=cfg.pipeline.max_seq_len, 
                    truncation=True, 
                    padding="max_length", 
                    return_tensors="pt"
                ).to(device)

                # Get clean activations from distilgpt2
                outputs = model(**inputs)
                clean_activations = outputs.hidden_states[cfg.pipeline.target_layer]

                # Apply experimental quantization damage
                test_activations = simulate_quantization(clean_activations, cfg.quantization)

                # Flatten for SAE
                flat_acts = test_activations.view(-1, cfg.pipeline.d_model)
                
                # Pass through SAE
                x_reconstructed, feature_acts, l2_loss = sae(flat_acts)

                # Calculate metrics
                l0_norm = (feature_acts > 0).float().sum(dim=-1).mean()
                variance_explained = 1.0 - (l2_loss / flat_acts.var(dim=0).mean())

                total_l2_loss += l2_loss.item()
                total_l0_norm += l0_norm.item()
                total_variance_explained += variance_explained.item()
                batches_processed += 1
                
                batch_texts = []

            if batches_processed >= 50: # Arbitrary cutoff for evaluation speed
                break

    # 5. Compile and Save Results
    results = {
        "quantization_bits": cfg.quantization.bits,
        "quantization_method": cfg.quantization.method,
        "average_l2_loss": total_l2_loss / batches_processed,
        "average_l0_norm": total_l0_norm / batches_processed,
        "average_variance_explained": total_variance_explained / batches_processed
    }

    print("\n--- Evaluation Results ---")
    for key, value in results.items():
        print(f"{key}: {value}")

    # Save to JSON for the report
    os.makedirs(cfg.output_dir, exist_ok=True)
    results_path = os.path.join(cfg.output_dir, f"eval_results_{cfg.quantization.bits}bit.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    evaluate_sae()