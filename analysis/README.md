# Degradation Curve Analysis

This module evaluates the robustness of the phishing/BEC detectors. It measures how the detectors' accuracy degrades as the adversarial perturbation budget increases from 1% to 100%.

The analysis runs an adversarial attack (like TextFooler) against a target detector and sweeps across different perturbation budgets to generate a degradation curve.

## Requirements

The project relies on the same Nix environment as the other modules. Enter the environment in this directory (or at the root if configured):

```bash
nix develop
```

*Note: Make sure the `detector_zoo` has a trained baseline model available, and that you have the dataset (`texts.json`) placed at the project root.*

## Running the Analysis

Run with default detector (`logistic_regression`):

```bash
python run_analysis.py
```

### Switching Detectors:

You can switch between any detector in `detector_zoo`:

```bash
# Logistic Regression (TF-IDF)
python run_analysis.py --detector logistic_regression

# Character-level CNN
python run_analysis.py --detector char_cnn

# Fine-tuned Transformer (DistilBERT)
python run_analysis.py --detector transformer

# Run sweep across all detectors sequentially:
python run_analysis.py --detector all
```

### Additional Options:

```bash
python run_analysis.py \
  --detector transformer \
  --attacker textfooler \
  --samples 10 \
  --max-budget 0.20 \
  --budget-steps 20 \
  --seed 42 \
  --report
```

*Note: Adding `--report` (or `--visualize`) automatically generates an interactive HTML dashboard in the results directory.*

### What it does:
1. Loads a random sample of malicious texts from the dataset.
2. Sweeps through perturbation budgets from 0.01 (1%) to 1.0 (100%).
3. Applies the adversarial attack to the samples for each budget.
4. Queries the detector to check if the attack successfully evaded detection (label flipped) or reduced the detector's confidence.
5. Saves results and generates a performance degradation curve.

## Generating Interactive HTML Reports

You can generate a visual HTML report at any time for any benchmark results folder:

```bash
# Generate report for a specific multi-detector or single-detector run:
python visualize_results.py results/20261012-021115

# Automatically pick the latest benchmark run:
python visualize_results.py

# Open report directly in default browser:
python visualize_results.py results/20261012-021115 --open

# Custom output file location:
python visualize_results.py results/20261012-021115 -o my_report.html
```

When running in the Nix environment from the project root:

```bash
nix develop . --command python analysis/visualize_results.py analysis/results/20261012-021115
```

### Report Features:
- **Git Diff Visualizer**: Toggle between Inline Word Diff (GitHub-style red/green), Side-by-Side Split Diff, Substitutions List, and Git Patch.
- **Outcome Status**: Clear indicators for defense pass/fail (Caught / Resisted vs Evaded / Label Flipped).
- **Interactive Degradation Curves**: Canvas-rendered degradation curves with $\pm 1\text{SE}$ detection bands and $\pm 1\sigma$ confidence bands, plus embedded matplotlib PNGs.
- **Multi-Detector Matrix**: Side-by-side comparison across CharCNN, Logistic Regression, and Transformer.
- **Sample Progression Heatmap**: Evasion timeline across all budget steps for every evaluated sample.
- **Dynamic Search & Filters**: Search text/substitutions, filter by outcome, budget step, sample index, or min confidence drop.
- **Raw Data Inspector**: Inspect full raw JSON/CSV records and export filtered datasets.

## Output

The script saves the following files into the `results/` directory:

- `report.html`: Self-contained interactive visualization dashboard.
- `raw_results.csv`: Contains the per-sample predictions and confidence scores for every budget step.
- `degradation_metrics.csv`: Aggregated detection rates, evasion rates, and mean confidence drops per budget step.
- `degradation_curve.png`: A plot showing the Detection Rate (%) and Mean Malicious Confidence (%) versus the Perturbation Budget (%).
