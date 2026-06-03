import torch
import hydra
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os

def simulate_quantization(tensor: torch.Tensor, config: DictConfig) -> torch.Tensor:
    """
    Applies simulated uniform affine quantization to the activation tensor.
    This degrades the mathematical precision of the tensor to simulate 8-bit or 4-bit hardware.
    """
    if config.bits >= 16 or config.method == "none":
        return tensor

    # Determine quantization boundaries based on bit-width
    q_min = 0 if not config.signed else -(2**(config.bits - 1))
    q_max = (2**config.bits - 1) if not config.signed else (2**(config.bits - 1) - 1)

    # Simple Min-Max calibration (Per-Tensor)
    t_min, t_max = tensor.min(), tensor.max()
    
    # Calculate Scale (S) and Zero-Point (Z)
    scale = (t_max - t_min) / (q_max - q_min)
    scale = torch.max(scale, torch.tensor(1e-8, device=tensor.device)) # Prevent div by zero
    
    zero_point = q_min - torch.round(t_min / scale)
    zero_point = torch.clamp(zero_point, q_min, q_max)

    # Quantize: Map float to integer steps
    q_tensor = torch.round(tensor / scale) + zero_point
    q_tensor = torch.clamp(q_tensor, q_min, q_max)

    # De-quantize: Map integer steps back to float space (with precision permanently lost)
    dq_tensor = (q_tensor - zero_point) * scale
    
    return dq_tensor

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def extract_activations(cfg: DictConfig):
    device = torch.device(cfg.pipeline.device if torch.cuda.is_available() else "cpu")
    print(f"Starting Extraction Pipeline on {device}...")

    # 1. Load Pre-trained Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_path)
    if cfg.data.add_bos_token:
        tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.pretrained_path,
        output_hidden_states=cfg.model.output_hidden_states,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    model.eval()

    # 2. Load Streaming Dataset
    dataset = load_dataset(
        cfg.data.name, 
        split=cfg.data.split, 
        streaming=cfg.data.streaming,
        trust_remote_code=True
    )

    # 3. Setup Extraction Storage
    os.makedirs(cfg.data.save_samples_dir, exist_ok=True)
    target_layer = cfg.pipeline.target_layer
    max_seq_len = cfg.pipeline.max_seq_len
    
    activation_buffer = []
    token_id_buffer = []
    token_buffer = []
    buffered_tokens = 0
    total_extracted = 0
    file_counter = 0
    flush_every = getattr(cfg.data, 'flush_every', 500_000)
    seq_len = max_seq_len
    
    print(f"Extracting activations from Layer {target_layer}. Target: {cfg.data.max_samples} tokens.")

    def flush_buffers(buffer, token_buffer_list, idx):
        stacked_acts = torch.cat(buffer, dim=0)
        stacked_tokens = torch.cat(token_buffer_list, dim=0)
        path = os.path.join(cfg.data.save_samples_dir, f"acts_{idx:04d}.pt")
        torch.save({
            "acts": stacked_acts.half(),
            "token_ids": stacked_tokens
        }, path)
        print(f"\nFlushed {stacked_acts.shape[0]:,} activations and {stacked_tokens.numel():,} token ids to {path}")
        return [], [], 0

    # 4. The Extraction Loop
    with torch.no_grad():
        for sample in tqdm(dataset, desc="Extracting"):
            ids = tokenizer(sample['text'], add_special_tokens=False)['input_ids']
            if not ids:
                continue

            token_buffer.extend(ids)
            while len(token_buffer) >= seq_len:
                chunk = token_buffer[:seq_len]
                token_buffer = token_buffer[seq_len:]
                inputs = torch.tensor([chunk], dtype=torch.long, device=device)

                outputs = model(inputs)
                target_activations = outputs.hidden_states[target_layer]

                # Normalize each token's feature vector before caching
                target_activations = target_activations / (target_activations.norm(dim=-1, keepdim=True) + 1e-8)

                flattened_acts = target_activations.view(-1, cfg.pipeline.d_model).cpu()
                flattened_tokens = torch.tensor(chunk, dtype=torch.long)

                activation_buffer.append(flattened_acts)
                token_id_buffer.append(flattened_tokens)
                buffered_tokens += seq_len
                total_extracted += seq_len

                if buffered_tokens >= flush_every:
                    activation_buffer, token_id_buffer, buffered_tokens = flush_buffers(
                        activation_buffer, token_id_buffer, file_counter
                    )
                    file_counter += 1

                if total_extracted >= cfg.data.max_samples:
                    break

            if total_extracted >= cfg.data.max_samples:
                break

    # Save any remaining activations in the buffer
    if activation_buffer:
        activation_buffer, token_id_buffer, buffered_tokens = flush_buffers(
            activation_buffer, token_id_buffer, file_counter
        )

    print("Extraction Complete.")

if __name__ == "__main__":
    extract_activations()