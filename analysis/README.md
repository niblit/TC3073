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

Run the main analysis script:

```bash
python run_analysis.py
```

### What it does:
1. Loads a random sample of malicious texts from the dataset.
2. Sweeps through perturbation budgets from 0.01 (1%) to 1.0 (100%).
3. Applies the adversarial attack to the samples for each budget.
4. Queries the detector to check if the attack successfully evaded detection (label flipped) or reduced the detector's confidence.
5. Saves results and generates a performance degradation curve.

## Output

The script saves the following files into the `results/` directory:

- `raw_results.csv`: Contains the per-sample predictions and confidence scores for every budget step.
- `degradation_metrics.csv`: Aggregated detection rates, evasion rates, and mean confidence drops per budget step.
- `degradation_curve.png`: A plot showing the Detection Rate (%) and Mean Malicious Confidence (%) versus the Perturbation Budget (%).
