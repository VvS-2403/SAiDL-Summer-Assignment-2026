import logging
import sys
import os

def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    """
    Creates a standardized logger that outputs cleanly formatted text to both 
    the console and an optional file.
    
    Args:
        name (str): The name of the logger (usually __name__ of the calling module).
        log_file (str, optional): The filepath to save the logs to.
        level (int): The logging threshold (e.g., logging.INFO, logging.DEBUG).
        
    Returns:
        logging.Logger: The configured logger instance.
    """
    # Create a standardized format: [YYYY-MM-DD HH:MM:SS] INFO - module_name: message
    formatter = logging.Formatter(
        fmt='[%(asctime)s] %(levelname)s - %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate log entries if this function is accidentally called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # 1. Console Handler: Prints to the terminal standard output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. File Handler: Writes to a permanent text file (if a path is provided)
    if log_file:
        # Ensure the directory exists before trying to write the file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger