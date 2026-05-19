# SAiDL Summer Assignment 2026

This repository contains the complete implementation for the SAiDL Summer Assignment 2026. It includes solutions for the compulsory **Core ML** track and the **Mechanistic Interpretability** domain-specific task.

## Repository Architecture

* **`core_ml/`**: Contains the custom Transformer architecture built from scratch. It includes implementations and benchmarking scripts for:
    * Attention Variants (Multi-Query, Grouped-Query, Linear, AFT-Local)
    * Positional Encodings (RoPE, ALiBi, Relative Bias)
    * Convolution + Attention Hybrids
* **`mechanistic_interpretability/`**: Contains the pipeline for analyzing `distilgpt2`. It includes:
    * Extraction of Layer 3 hidden states using streaming datasets
    * A custom Top-k Sparse Autoencoder (SAE) implementation
    * Per-tensor and per-feature 8-bit/4-bit quantization scripts
    * Spectral analysis and geometric evaluation tools
* **`reports/`**: Contains the LaTeX source code and generated PDFs detailing the methodology, empirical results, and mechanistic explanations.

## Local Setup & Installation

This repository is designed to be highly modular and relies on Hydra for configuration management.

1.  Ensure you have a Python 3.10+ environment active.
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Data & Weights Note**: This project utilizes `WikiText-2` for the Core ML baselines and a 1% subset of `OpenWebText` alongside `distilgpt2` weights for Interpretability. Ensure these are cached locally if running in an offline environment.

## Execution

*(Detailed execution commands for training loops and benchmarking scripts will be added here prior to final submission).*