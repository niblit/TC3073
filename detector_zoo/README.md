# Detector Zoo

This repository contains the **Detector Zoo** component of the reproducible evaluation engine, designed to stress-test phishing and Business Email Compromise (BEC) detectors.

The implementation provides a **detector zoo** with three interchangeable models, all sharing a uniform `fit`/`predict`/`save`/`load` interface via a common `BaseDetector` ABC:

| Detector | Module | Description |
|---|---|---|
| **Baseline** | `src/model_pipeline.py` | TF-IDF + Logistic Regression (scikit-learn) |
| **CharCNN** | `src/char_cnn.py` | Character-level CNN with multi-kernel 1D convolutions (PyTorch) |
| **Transformer** | `src/transformer_detector.py` | Fine-tuned DistilBERT (Hugging Face Transformers) |

## Requirements

The project uses Nix to ensure a reproducible execution environment, with `uv` for fast Python package management.
- [Nix](https://nixos.org/download.html) (with flakes enabled)
- (Fallback) Python 3.12+ and `pip install -r requirements.txt`

## Getting Started

1. **Enter the Reproducible Environment**
   Run the following command to enter the Nix shell, which provides Python 3.12 and automatically installs all dependencies via `uv`:
   ```bash
   nix develop
   ```
   On first run this creates a `.venv` and installs all packages from `requirements.txt`.

2. **Dataset**
   Ensure your dataset (`combined_full.json`) is in the root directory. The dataset should be a list of JSON objects with `"text"` and `"label"` keys. (0 for safe, 1 for malicious).

3. **Run the Tests**
   Inside the nix shell, you can run the full test suite using `pytest`:
   ```bash
   pytest
   ```

4. **Train All Detectors**
   Run the main script to load the data, split it with fixed seeds, train all three detectors, and evaluate them:
   ```bash
   python main.py
   ```
   This will output metrics for each detector and save reports to `reports/` and models to `models/`.

5. **Train Specific Detectors**
   You can select which detectors to train:
   ```bash
   python main.py --detectors baseline charcnn
   python main.py --detectors transformer
   ```

6. **Predict with Any Detector**
   ```bash
   echo "Click here to claim your prize" | python predict.py --type baseline
   echo "Click here to claim your prize" | python predict.py --type charcnn
   echo "Click here to claim your prize" | python predict.py --type transformer
   ```

## Architecture Overview

- `src/base_detector.py`: Abstract base class defining the common detector interface (`fit`, `predict`, `predict_proba`, `save_model`, `load_model`).
- `src/model_pipeline.py`: Baseline detector (TF-IDF + Logistic Regression).
- `src/char_cnn.py`: Character-level CNN detector (PyTorch).
- `src/transformer_detector.py`: Fine-tuned DistilBERT detector (Hugging Face).
- `src/data_loader.py`: Handles parsing and validation of the JSON dataset.
- `src/splitter.py`: Scikit-learn wrapper that guarantees reproducible train/test splits.
- `src/metrics.py`: Computes Accuracy, Precision, Recall, and F1-Score, and generates text reports.
- `main.py`: The main entry point — trains, evaluates, and saves all detectors.
- `predict.py`: CLI for inference with any trained detector.
- `tests/`: Contains isolated pytest unit tests for every module and detector.

## Extensibility

To add a new detector, create a class that inherits from `BaseDetector` in `src/base_detector.py`, implement the required methods, and register it in the `DETECTOR_ZOO` dict in `main.py`.
