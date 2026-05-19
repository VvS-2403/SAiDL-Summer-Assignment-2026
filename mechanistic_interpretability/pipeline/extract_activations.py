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
        output_hidden_states=cfg.model.output_hidden_states
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
    batch_size = cfg.pipeline.batch_size
    max_seq_len = cfg.pipeline.max_seq_len
    
    activation_buffer = []
    total_extracted = 0
    file_counter = 0
    
    print(f"Extracting activations from Layer {target_layer}. Target: {cfg.data.max_samples} tokens.")

    # 4. The Extraction Loop
    with torch.no_grad():
        batch_texts = []
        for sample in tqdm(dataset):
            batch_texts.append(sample['text'])
            
            # Once we have enough texts, process the batch
            if len(batch_texts) == batch_size:
                inputs = tokenizer(
                    batch_texts, 
                    max_length=max_seq_len, 
                    truncation=True, 
                    padding="max_length", 
                    return_tensors="pt"
                ).to(device)

                # Forward pass: model returns (logits, past_key_values, hidden_states)
                outputs = model(**inputs)
                
                # hidden_states is a tuple of (embedding_out, layer1_out, layer2_out, ...)
                # Extract the specific layer we want
                target_activations = outputs.hidden_states[target_layer]

                # Apply experimental Quantization damage
                damaged_activations = simulate_quantization(target_activations, cfg.quantization)

                # Flatten the batch and sequence dimensions: (batch * seq_len, d_model)
                flattened_acts = damaged_activations.view(-1, cfg.pipeline.d_model)
                
                # Move to CPU RAM immediately to prevent GPU OOM
                activation_buffer.append(flattened_acts.cpu())
                total_extracted += flattened_acts.shape[0]
                batch_texts = []

            # 5. Disk Flushing (Cache to hard drive when buffer gets too large)
            if total_extracted >= 1_000_000:
                stacked_buffer = torch.cat(activation_buffer, dim=0)
                file_path = os.path.join(cfg.data.save_samples_dir, f"acts_{file_counter}.pt")
                torch.save(stacked_buffer, file_path)
                print(f"\nFlushed {stacked_buffer.shape[0]} activations to {file_path}")
                
                activation_buffer = []
                total_extracted = 0
                file_counter += 1

            if file_counter * 1_000_000 >= cfg.data.max_samples:
                break

    # Save any remaining activations in the buffer
    if activation_buffer:
        stacked_buffer = torch.cat(activation_buffer, dim=0)
        file_path = os.path.join(cfg.data.save_samples_dir, f"acts_{file_counter}.pt")
        torch.save(stacked_buffer, file_path)
        
    print("Extraction Complete.")

if __name__ == "__main__":
    extract_activations()