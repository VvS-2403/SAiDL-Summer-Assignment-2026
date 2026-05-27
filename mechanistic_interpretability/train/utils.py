import torch

from mechanistic_interpretability.models.sae import TopKSparseAutoencoder


def load_sae(path: str, d_model: int, expansion_factor: int, k: int, device: str = 'cpu') -> TopKSparseAutoencoder:
    sae = TopKSparseAutoencoder(d_model=d_model, expansion_factor=expansion_factor, k=k)
    sae.load_state_dict(torch.load(path, map_location=device))
    sae.eval()
    return sae.to(device)


def get_feature_activations(
    sae: TopKSparseAutoencoder,
    activations: torch.Tensor,
    device: str = 'cpu',
    batch_size: int = 2048,
) -> torch.Tensor:
    sae.eval()
    outputs = []
    with torch.no_grad():
        for i in range(0, activations.size(0), batch_size):
            chunk = activations[i : i + batch_size].to(device)
            _, feats, _ = sae(chunk)
            outputs.append(feats.cpu())
    return torch.cat(outputs, dim=0)
