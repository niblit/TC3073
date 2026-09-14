# Evasion and Robustness Engine for Phishing/BEC Detectors

This repository contains a reproducible, extensible evaluation engine designed to stress-test phishing and Business Email Compromise (BEC) detectors under explicit perturbation budgets.

The current implementation provides a clean baseline using a classic machine learning model (TF-IDF + Logistic Regression). It is built with a highly modular architecture and strict Test-Driven Development (TDD).

## Requirements

The project uses Nix to ensure a reproducible execution environment.
- [Nix](https://nixos.org/download.html) (with flakes enabled)
- (Fallback) Python 3.12+ and `pip install -r requirements.txt`

## Getting Started

1. **Enter the Reproducible Environment**
   Run the following command to enter the Nix shell which provides Python 3.12 and all required data science libraries:
   ```bash
   nix develop
   ```

2. **Dataset**
   Ensure your dataset (`combined_full.json`) is in the root directory. The dataset should be a list of JSON objects with `"text"` and `"label"` keys. (0 for safe, 1 for malicious).

3. **Run the Tests**
   Inside the nix shell, you can run the full test suite using `pytest`:
   ```bash
   pytest
   ```

4. **Execute the Baseline Pipeline**
   Run the main script to load the data, split it with fixed seeds, train the baseline model, and evaluate it:
   ```bash
   python main.py
   ```
   This will output the metrics and save a detailed report to `reports/baseline_report.txt`.

## Architecture Overview

- `src/data_loader.py`: Handles parsing and validation of the JSON dataset.
- `src/splitter.py`: Scikit-learn wrapper that guarantees reproducible train/test splits.
- `src/model_pipeline.py`: Definition of the model (currently TF-IDF + Logistic Regression).
- `src/metrics.py`: Computes Accuracy, Precision, Recall, and F1-Score, and generates the text report.
- `main.py`: The main entry point coordinating the data flow and execution.
- `tests/`: Contains isolated pytest unit tests for every module.

## Extensibility

This architecture is designed to easily accommodate new models (e.g., Character-Level CNNs via PyTorch or fine-tuned transformers via Hugging Face). You can swap out the pipeline in `main.py` with any model that adheres to a standard `fit`/`predict` interface.
