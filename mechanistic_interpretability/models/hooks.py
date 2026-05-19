import torch
import torch.nn as nn
from typing import Dict, List, Callable, Union

class ActivationCache:
    """
    A context manager and utility class designed to surgically wiretap a PyTorch model.
    It uses forward hooks to intercept and cache intermediate tensor activations 
    during a forward pass without modifying the original model's source code.
    """
    def __init__(self):
        self.cache: Dict[str, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _generate_caching_hook(self, layer_name: str) -> Callable:
        """
        Generates a PyTorch forward hook function.
        When the target layer finishes its math, this hook catches the output tensor,
        detaches it from the gradient graph, and stores it in the cache dictionary.
        """
        def hook(module: nn.Module, inputs: tuple, output: Union[torch.Tensor, tuple]):
            # HuggingFace models often return tuples (hidden_state, past_key_values)
            # We strictly want the hidden_state activation tensor.
            if isinstance(output, tuple):
                activation = output[0]
            else:
                activation = output
                
            # Detach from computational graph to prevent memory leaks,
            # and optionally move to CPU to save fragile GPU VRAM.
            self.cache[layer_name] = activation.detach().cpu()
            
        return hook

    def register(self, model: nn.Module, target_layers: List[str]):
        """
        Traverses the model's architecture and attaches the caching hooks to the specified layers.
        
        Args:
            model (nn.Module): The target neural network (e.g., distilgpt2 or your custom core_ml model).
            target_layers (List[str]): Exact string names of the sub-modules to wiretap.
        """
        for name, module in model.named_modules():
            if name in target_layers:
                # register_forward_hook returns a handle that can be used to remove the hook later
                handle = module.register_forward_hook(self._generate_caching_hook(name))
                self.hooks.append(handle)
                
        if not self.hooks:
            print(f"Warning: No hooks registered. Ensure layer names {target_layers} match the model architecture.")

    def clear_cache(self):
        """Empties the stored tensors to free up system RAM."""
        self.cache.clear()

    def remove_hooks(self):
        """Detaches all physical hooks from the PyTorch model."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    # --- Context Manager Magic Methods ---
    # These allow the class to be used cleanly in a 'with' statement.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()
        self.clear_cache()