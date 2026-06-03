import json
import math
import os

import hydra
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from mechanistic_interpretability.models.quantizer import TensorQuantizer
from mechanistic_interpretability.models.sae import TopKSparseAutoencoder
from mechanistic_interpretability.utils.metrics import linear_cka, compute_sds, variance_explained


def make_replacement_hook(replacement_tensor):
    def hook(module, input_, output):
        if isinstance(output, tuple):
            return (replacement_tensor,) + output[1:]
        return replacement_tensor
    return hook


def compute_perplexity_with_quantised_layer3(model, tokenizer, text_batch, quant_fn, device, target_layer=3):
    import torch.nn.functional as F
    inputs = tokenizer(text_batch, return_tensors="pt", truncation=True, max_length=128, padding=True).to(device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        out_clean = model(**inputs, output_hidden_states=True)
    h3_clean = out_clean.hidden_states[target_layer].clone()
    h3_quant = quant_fn(h3_clean)

    handle = model.transformer.h[target_layer - 1].register_forward_hook(
        make_replacement_hook(h3_quant)
    )
    with torch.no_grad():
        out_quant = model(**inputs)
    handle.remove()

    logits = out_quant.logits
    shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
    shift_labels = input_ids[:, 1:].reshape(-1)
    loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=tokenizer.pad_token_id or -100)
    return loss.item()


def quantise_tensor(h: torch.Tensor, bits: int, mode: str = "per_tensor") -> torch.Tensor:
    q_min = -(2 ** (bits - 1))
    q_max = 2 ** (bits - 1) - 1
    if mode == "per_tensor":
        t_min, t_max = h.min(), h.max()
        scale = (t_max - t_min).clamp(min=1e-8) / (q_max - q_min)
        zp = (q_min - (t_min / scale).round()).clamp(q_min, q_max)
    else:
        t_min = h.reshape(-1, h.shape[-1]).min(0).values
        t_max = h.reshape(-1, h.shape[-1]).max(0).values
        t_min = t_min.view(*([1] * (h.dim() - 1)), -1)
        t_max = t_max.view(*([1] * (h.dim() - 1)), -1)
        scale = (t_max - t_min).clamp(min=1e-8) / (q_max - q_min)
        zp = (q_min - (t_min / scale).round()).clamp(q_min, q_max)
    q = (h / scale + zp).round().clamp(q_min, q_max)
    return (q - zp) * scale


def compute_sweep_metrics(clean: torch.Tensor, quant: torch.Tensor, bits: int, mode: str):
    flat_clean = clean.view(-1, clean.shape[-1]).cpu().numpy()
    flat_quant = quant.view(-1, quant.shape[-1]).cpu().numpy()
    mse = float(((clean - quant) ** 2).mean().item())
    sds = compute_sds(flat_clean, flat_quant, k=min(32, clean.shape[-1]))
    cka = linear_cka(flat_clean[:1000], flat_quant[:1000])
    return mse, sds, cka


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
    quant_modes = ["per_tensor", "per_feature"]
    results = {}

    for bits in bit_widths:
        for mode in quant_modes:
            key = f"{bits}bit_{mode}"
            print(f"\nRunning sweep: {key}")
            total_mse = 0.0
            total_sds = 0.0
            total_cka = 0.0
            total_ppl = 0.0
            count = 0
            batch_texts = []
            sample_texts = []

            for sample in dataset:
                batch_texts.append(sample["text"])
                sample_texts.append(sample["text"])
                if len(batch_texts) < cfg.pipeline.batch_size:
                    continue

                inputs = tokenizer(
                    batch_texts,
                    max_length=cfg.pipeline.max_seq_len,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    clean = outputs.hidden_states[cfg.pipeline.target_layer]
                    quant = quantise_tensor(clean, bits, mode)
                    mse, sds, cka = compute_sweep_metrics(clean, quant, bits, mode)

                total_mse += mse
                total_sds += sds
                total_cka += cka

                for text in sample_texts[:5]:
                    total_ppl += compute_perplexity_with_quantised_layer3(
                        model, tokenizer, [text], lambda h: quantise_tensor(h, bits, mode), device
                    )
                    count += 1
                    if count >= 10:
                        break

                batch_texts = []
                sample_texts = []
                if count >= cfg.pipeline.get("max_batches", 10):
                    break

            results[key] = {
                "bits": bits,
                "mode": mode,
                "avg_mse": total_mse / max(1, count),
                "avg_sds": total_sds / max(1, count),
                "avg_cka": total_cka / max(1, count),
                "avg_ppl": math.exp(total_ppl / max(1, count)),
            }

    os.makedirs(cfg.output_dir, exist_ok=True)
    out_path = os.path.join(cfg.output_dir, "quantization_sweep.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved quantization results to {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    quantize_sweep()
