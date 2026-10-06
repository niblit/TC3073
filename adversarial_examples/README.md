# Adversarial Examples Engine

This module generates adversarial examples to evade phishing and Business Email Compromise (BEC) detectors. It is designed around an explicit perturbation budget to ensure the generated texts preserve their original malicious intent and remain realistic.

Currently, it implements the **TextFooler** algorithm (Jin et al., 2019), a strong baseline for black-box natural language attacks.

## Architecture

- `harness/taxonomy.py`: Defines the base transformations (`TargetedContentTransformation`, `StructureTransformation`, `MetadataTransformation`).
- `harness/content_attacks.py`: Contains the `TextFoolerAttack` implementation (Word Importance Ranking, Synonym Extraction, POS & USE Filtering, Iterative Attack Loop).
- `harness/budget.py`: Tracks and enforces the perturbation budget (`max_edits`).
- `harness/nlp_loader.py`: Lazy-loads heavy NLP models (spaCy, Universal Sentence Encoder, Counter-fitted Word Vectors).
- `main.py`: The CLI entry point to apply an attack to an input text using a specific target model oracle.

## Requirements

The project uses Nix to ensure a reproducible execution environment, with `uv` for fast Python package management.
- [Nix](https://nixos.org/download.html) (with flakes enabled)

Counter-fitted word vectors are required for the TextFooler algorithm. Download them and place them at `data/counter-fitted-vectors.txt` (or set `COUNTER_FITTED_VECTORS_PATH`).

## Getting Started

1. **Enter the Reproducible Environment**
   Run the following command to enter the Nix shell:
   ```bash
   nix develop
   ```
   This creates a `.venv`, installs dependencies (like TensorFlow, gensim, spaCy), and downloads required NLTK datasets.

2. **Generate an Adversarial Example**
   You can run `main.py` passing input text and a budget (as a percentage of total words, e.g., 0.1 for 10%). It requires a target model oracle command to evaluate predictions during the attack:

   ```bash
   echo "Click here to update your account password immediately." | python main.py --budget 0.2
   ```
   *Note: By default, the `--oracle-cmd` points to the baseline detector in `../detector_zoo`.*

3. **Run the Tests**
   Inside the nix shell, you can run the comprehensive test suite using `pytest`:
   ```bash
   pytest
   ```
   All tests are fully mocked to run deterministically and blazingly fast without requiring the heavy NLP models or the actual counter-fitted vector file.
