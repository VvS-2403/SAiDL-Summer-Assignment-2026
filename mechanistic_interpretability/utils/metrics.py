import torch
import numpy as np


def compute_l2_change(original: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
    """Compute the average L2 change per feature dimension."""
    return (original - quantized).pow(2).mean(dim=0).sqrt()


def _soft_histogram(x: torch.Tensor, bins: int = 64) -> torch.Tensor:
    d = x.shape[1]
    mins = x.min(dim=0).values
    maxs = x.max(dim=0).values
    hist = []
    for i in range(d):
        hist.append(torch.histc(x[:, i].float(), bins=bins, min=mins[i].item(), max=maxs[i].item()))
    return torch.stack(hist, dim=0)


def compute_kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p_hist = _soft_histogram(p) + eps
    q_hist = _soft_histogram(q) + eps
    p_hist = p_hist / p_hist.sum(dim=1, keepdim=True)
    q_hist = q_hist / q_hist.sum(dim=1, keepdim=True)
    return (p_hist * (p_hist / q_hist).log()).sum(dim=1)


def variance_explained(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    mse = (original - reconstructed).pow(2).mean()
    var = original.var()
    if var < 1e-8:
        return 0.0
    return float(1.0 - mse / var)


def compute_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    K = X @ X.T
    L = Y @ Y.T
    hsic = _hsic(K, L)
    denom = np.sqrt(_hsic(K, K) * _hsic(L, L))
    return float(hsic / denom) if denom > 0 else 0.0


def _hsic(K: np.ndarray, L: np.ndarray) -> float:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return float(np.trace(K @ H @ L @ H)) / ((n - 1) ** 2)


def compute_sds(clean: np.ndarray, quant: np.ndarray) -> float:
    n = min(len(clean), len(quant))
    clean = clean[:n]
    quant = quant[:n]
    norm = np.linalg.norm(clean)
    if norm < 1e-10:
        return 0.0
    return float(np.linalg.norm(clean - quant) / norm)
