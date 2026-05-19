import torch
import torch.nn as nn

class TensorQuantizer:
    """
    A utility class for simulating N-bit integer quantization on PyTorch tensors.
    This performs 'Fake Quantization' (squashing to discrete steps but returning float32)
    to allow standard operations and gradient flow without requiring custom CUDA kernels.
    """
    def __init__(self, bits: int = 8, method: str = "uniform_affine", signed: bool = True, per_channel: bool = False):
        """
        Args:
            bits (int): Target bit-width (e.g., 8 for INT8, 4 for INT4).
            method (str): "uniform_affine" (asymmetric) or "uniform_symmetric" (symmetric around 0).
            signed (bool): If True, maps to [-2^(b-1), 2^(b-1)-1]. If False, maps to [0, 2^b-1].
            per_channel (bool): If True, calculates scale/zero-point per row/column instead of globally.
        """
        self.bits = bits
        self.method = method
        self.signed = signed
        self.per_channel = per_channel

        # Define the absolute clipping boundaries based on the requested bit-width
        if self.signed:
            self.q_min = -(2**(self.bits - 1))
            self.q_max = (2**(self.bits - 1)) - 1
        else:
            self.q_min = 0
            self.q_max = (2**self.bits) - 1

    def _compute_scale_and_zeropoint(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calibrates the mathematical mapping parameters based on the tensor's absolute range.
        """
        if self.per_channel:
            # Assume dimension 0 is the output channel (standard for nn.Linear weights)
            # This prevents outlier features from destroying the resolution of standard features
            t_min = tensor.amin(dim=1, keepdim=True)
            t_max = tensor.amax(dim=1, keepdim=True)
        else:
            # Global min/max for the entire tensor block
            t_min = tensor.min()
            t_max = tensor.max()

        if self.method == "uniform_symmetric":
            # Force the range to be perfectly symmetric around exactly 0.0
            abs_max = torch.max(t_min.abs(), t_max.abs())
            scale = abs_max / self.q_max
            zero_point = torch.zeros_like(scale)
        else: 
            # uniform_affine (Asymmetric)
            scale = (t_max - t_min) / (self.q_max - self.q_min)
            # Prevent division by zero if the tensor is entirely uniform (e.g., all 0s)
            scale = torch.clamp(scale, min=1e-8)
            zero_point = self.q_min - torch.round(t_min / scale)
            zero_point = torch.clamp(zero_point, self.q_min, self.q_max)

        # Prevent global division by zero
        scale = torch.clamp(scale, min=1e-8)
        return scale, zero_point

    def quantize(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compresses the float32 tensor down to the discrete N-bit integer steps.
        """
        # Bypass quantization entirely if baseline (32-bit) is requested
        if self.bits >= 32 or self.method == "none":
            return tensor, torch.tensor(1.0, device=tensor.device), torch.tensor(0.0, device=tensor.device)

        scale, zero_point = self._compute_scale_and_zeropoint(tensor)
        
        # 1. Scale down and shift the floating point values
        q_tensor = torch.round(tensor / scale) + zero_point
        # 2. Hard clamp to the strict integer boundaries (e.g., -128 to 127)
        q_tensor = torch.clamp(q_tensor, self.q_min, self.q_max)
        
        return q_tensor, scale, zero_point

    def dequantize(self, q_tensor: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
        """
        Unpacks the compressed tensor back into float32 space.
        Note: The fine-grained mathematical precision is permanently lost during this step.
        """
        if self.bits >= 32 or self.method == "none":
            return q_tensor
            
        dq_tensor = (q_tensor - zero_point) * scale
        return dq_tensor

    def simulate(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs Fake Quantization in a single seamless step (Quantize -> Dequantize).
        This is the primary method called during your ablation studies in evaluate.py.
        """
        q_tensor, scale, zero_point = self.quantize(tensor)
        return self.dequantize(q_tensor, scale, zero_point)