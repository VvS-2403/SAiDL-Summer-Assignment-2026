import torch
import math

def calculate_perplexity(loss: float) -> float:
    """
    Calculates the perplexity of a language model given its cross-entropy loss.
    
    Args:
        loss (float): The average Cross-Entropy loss over a batch or epoch.
        
    Returns:
        float: The perplexity score. Returns float('inf') if loss is too high to exponentiate safely.
    """
    try:
        return math.exp(loss)
    except OverflowError:
        return float('inf')

def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> float:
    """
    Computes the exact token-matching accuracy.
    While perplexity is the standard for language models, accuracy provides
    a more interpretable metric for absolute prediction correctness.
    
    Args:
        logits (torch.Tensor): The unnormalized predictions of shape (batch * seq_len, vocab_size).
        targets (torch.Tensor): The ground truth tokens of shape (batch * seq_len).
        ignore_index (int): The padding index to ignore in the calculation.
        
    Returns:
        float: The percentage of correctly predicted tokens (0.0 to 1.0).
    """
    # Find the vocabulary index with the highest probability
    predictions = torch.argmax(logits, dim=-1)
    
    # Create a mask to ignore padding tokens (if any exist)
    mask = targets != ignore_index
    
    # Calculate how many predictions match the targets within the unmasked regions
    correct = (predictions == targets) & mask
    
    # Calculate average accuracy
    if mask.sum().item() == 0:
        return 0.0
        
    accuracy = correct.sum().float() / mask.sum().float()
    return accuracy.item()