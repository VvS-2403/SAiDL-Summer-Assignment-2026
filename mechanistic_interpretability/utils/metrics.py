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


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    Kx = X @ X.T
    Ky = Y @ Y.T

    def centre(K: np.ndarray) -> np.ndarray:
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    Kx_c = centre(Kx)
    Ky_c = centre(Ky)

    hsic_xy = np.sum(Kx_c * Ky_c)
    hsic_xx = np.sqrt(np.sum(Kx_c * Kx_c))
    hsic_yy = np.sqrt(np.sum(Ky_c * Ky_c))
    return float(hsic_xy / (hsic_xx * hsic_yy + 1e-8))


def compute_sds(clean: np.ndarray, quant: np.ndarray, k: int = 32) -> float:
    """Subspace Distance Score (SDS) between clean and quantized activations."""
    n = min(clean.shape[0], quant.shape[0])
    if n == 0:
        return 0.0
    H_c = clean[:n] - clean[:n].mean(axis=0, keepdims=True)
    H_q = quant[:n] - quant[:n].mean(axis=0, keepdims=True)
    _, _, Vc = np.linalg.svd(H_c, full_matrices=False)
    _, _, Vq = np.linalg.svd(H_q, full_matrices=False)
    k = min(k, Vc.shape[0], Vq.shape[0])
    Vc_k = Vc[:k].T
    Vq_k = Vq[:k].T
    M = Vc_k.T @ Vq_k
    sv = np.linalg.svd(M, compute_uv=False)
    cos_angles = np.clip(sv, -1.0, 1.0)
    return float(1.0 - cos_angles.mean())
