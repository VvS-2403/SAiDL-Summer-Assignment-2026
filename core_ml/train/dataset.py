import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import logging

# Import the base PyTorch Dataset class we wrote earlier
from core_ml.data.dataset import LanguageModelingDataset

logger = logging.getLogger(__name__)

def prepare_dataloaders(config):
    """
    Orchestrates the entire data pipeline: downloading, tokenizing, chunking, and batching.
    
    Args:
        config: The Hydra configuration object containing dataset parameters.
    Returns:
        train_loader, val_loader: PyTorch DataLoaders ready for the training loop.
    """
    logger.info(f"Loading {config.dataset.name} ({config.dataset.config_name})...")
    
    # 1. Load the raw text dataset from HuggingFace
    raw_datasets = load_dataset(config.dataset.name, config.dataset.config_name)
    
    # 2. Initialize the Tokenizer
    # We strictly use the distilgpt2 tokenizer to align with the Interpretability task later
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    
    def tokenize_function(examples):
        # Extracts the 'text' column and converts strings to integer IDs
        return tokenizer(examples["text"], truncation=False, return_attention_mask=False)

    logger.info("Tokenizing dataset...")
    # Map the tokenizer across the dataset in batches for massive speedups
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Running tokenizer on dataset"
    )

    # 3. Flatten the dataset into continuous 1D streams
    # HuggingFace returns lists of lists; we need one massive flat list
    train_stream = [token for sublist in tokenized_datasets["train"]["input_ids"] for token in sublist]
    val_stream = [token for sublist in tokenized_datasets["validation"]["input_ids"] for token in sublist]
    
    # 4. Wrap into our custom PyTorch Dataset (from core_ml/data/dataset.py)
    train_dataset = LanguageModelingDataset(train_stream, seq_len=config.dataset.seq_len)
    val_dataset = LanguageModelingDataset(val_stream, seq_len=config.dataset.seq_len)
    
    # 5. Create the DataLoaders
    # Pin_memory accelerates CPU-to-GPU data transfers
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.dataset.batch_size, 
        shuffle=True, 
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.dataset.batch_size, 
        shuffle=False, 
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
        drop_last=True
    )
    
    logger.info(f"Dataloaders ready. Train batches: {len(train_loader)}. Val batches: {len(val_loader)}.")
    return train_loader, val_loader