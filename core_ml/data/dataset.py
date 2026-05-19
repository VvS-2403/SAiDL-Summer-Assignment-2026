import torch
from torch.utils.data import Dataset

class LanguageModelingDataset(Dataset):
    """
    A standard PyTorch Dataset for autoregressive language modeling.
    Takes a continuous stream of tokenized text and chunks it into fixed-length sequences.
    """
    def __init__(self, tokenized_data, seq_len):
        """
        Args:
            tokenized_data (list or torch.Tensor): A flat 1D sequence of integer token IDs.
            seq_len (int): The maximum context window for the model (e.g., 1024).
        """
        # Ensure the data is a 1D PyTorch tensor
        if not isinstance(tokenized_data, torch.Tensor):
            self.tokens = torch.tensor(tokenized_data, dtype=torch.long)
        else:
            self.tokens = tokenized_data.flatten()
            
        self.seq_len = seq_len
        
        # We need seq_len + 1 tokens for a single (input, target) pair.
        # This calculates how many non-overlapping chunks we can cleanly extract.
        self.num_samples = (len(self.tokens) - 1) // self.seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Fetches the input sequence and the shifted target sequence.
        """
        # Calculate the starting index for this specific chunk
        start_idx = idx * self.seq_len
        
        # We grab exactly seq_len + 1 tokens
        end_idx = start_idx + self.seq_len + 1
        
        chunk = self.tokens[start_idx:end_idx]
        
        # Inputs (x): All tokens except the very last one
        x = chunk[:-1]
        
        # Targets (y): All tokens except the very first one (shifted by 1)
        y = chunk[1:]
        
        return x, y
    
    