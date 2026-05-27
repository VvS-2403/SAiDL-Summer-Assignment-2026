import json
import os

import hydra
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from mechanistic_interpretability.models.quantizer import TensorQuantizer
from mechanistic_interpretability.models.sae import TopKSparseAutoencoder
from mechanistic_interpretability.utils.metrics import variance_explained


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def quantize_sweep(cfg: DictConfig):
    device = torch.device(cfg.pipeline.device if torch.cuda.is_available() else "cpu")
    print(f"Starting quantization sweep on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.pretrained_path,
        output_hidden_states=True,
    ).to(device)
    model.eval()

    sae_path = os.path.join(cfg.output_dir, "sae_final.pt")
    sae = TopKSparseAutoencoder(
        d_model=cfg.pipeline.d_model,
        expansion_factor=cfg.sae.expansion_factor,
        k=cfg.sae.k,
    ).to(device)
    sae.load_state_dict(torch.load(sae_path, map_location=device))
    sae.eval()

    dataset = load_dataset(
        cfg.data.name,
        split=cfg.data.split,
        streaming=cfg.data.streaming,
        trust_remote_code=True,
    )

    bit_widths = [8, 4, 2]
    results = {}
    for bits in bit_widths:
        quantizer = TensorQuantizer(
            bits=bits,
            method=cfg.quantization.method,
            signed=cfg.quantization.signed,
            per_channel=cfg.quantization.per_channel,
        )
        total_var_exp = 0.0
        total_l2 = 0.0
        count = 0
        batch_texts = []

        print(f"\nRunning {bits}-bit sweep...")
        for sample in tqdm(dataset, total=cfg.data.get("max_samples", 200) // cfg.pipeline.batch_size):
            batch_texts.append(sample["text"])
            if len(batch_texts) < cfg.pipeline.batch_size:
                continue

            inputs = tokenizer(
                batch_texts,
                max_length=cfg.pipeline.max_seq_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                clean = outputs.hidden_states[cfg.pipeline.target_layer]
                quant = quantizer.simulate(clean)

            flat_clean = clean.view(-1, cfg.pipeline.d_model)
            flat_quant = quant.view(-1, cfg.pipeline.d_model)

            total_var_exp += variance_explained(flat_clean, flat_quant)
            total_l2 += (flat_clean - flat_quant).pow(2).mean().item()
            count += 1
            batch_texts = []
            if count >= cfg.pipeline.get("max_batches", 50):
                break

        results[f"{bits}bit"] = {
            "bits": bits,
            "avg_variance_explained": total_var_exp / max(1, count),
            "avg_l2": total_l2 / max(1, count),
        }

    os.makedirs(cfg.output_dir, exist_ok=True)
    out_path = os.path.join(cfg.output_dir, "quantization_sweep.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved quantization results to {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    quantize_sweep()
