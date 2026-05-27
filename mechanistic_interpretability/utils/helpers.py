import os
import random
import torch
import numpy as np


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_tensor(tensor: torch.Tensor, path: str):
    ensure_dir(os.path.dirname(path))
    torch.save(tensor, path)


def load_tensor(path: str, device: str = "cpu") -> torch.Tensor:
    return torch.load(path, map_location=device)
