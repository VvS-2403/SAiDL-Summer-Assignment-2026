import torch
import random
import numpy as np
import os

def set_seed(seed: int = 42):
    """
    Sets the seed for all random number generators to ensure reproducibility.
    Crucial for benchmarking different attention architectures fairly.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU environments
        
        # Enforce deterministic CuDNN algorithms (might slightly reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def count_parameters(model: torch.nn.Module) -> int:
    """
    Returns the total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(model, optimizer, scheduler, epoch, global_step, val_loss, filepath):
    """
    Saves a comprehensive checkpoint of the training state.
    Allows for strict resumption if a local training run is interrupted.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_loss': val_loss
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device='cpu'):
    """
    Loads a model (and optionally optimizer/scheduler) from a checkpoint.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No checkpoint found at {filepath}")
        
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    print(f"Successfully loaded checkpoint from {filepath} (Epoch {checkpoint.get('epoch', 'N/A')})")
    return checkpoint.get('global_step', 0), checkpoint.get('val_loss', float('inf'))