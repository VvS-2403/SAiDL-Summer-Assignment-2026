import torch
from torch.utils.data import Dataset


class ActivationDataset(Dataset):
    """
    Wraps a flat activation tensor of shape (N, d_model) for training and evaluation.
    """
    def __init__(self, activations: torch.Tensor):
        assert activations.ndim == 2, "Expected activations with shape (N, d_model)"
        self.activations = activations.float()

    def __len__(self):
        return self.activations.size(0)

    def __getitem__(self, idx):
        return self.activations[idx]
