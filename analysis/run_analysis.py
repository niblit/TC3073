#!/usr/bin/env python3
"""
Degradation Curve Analysis
===========================
Measures how a detector's accuracy degrades as the adversarial
perturbation budget increases from 1% to 100%.

Usage (from within `nix develop`):
    python run_analysis.py
"""

import sys
import os
import json
import random
import csv
import warnings
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Silence noisy libraries BEFORE any imports from them
# ---------------------------------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Add sibling project directories to sys.path so we can import their modules
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "detector_zoo"))
sys.path.insert(0, str(PROJECT_ROOT / "adversarial_examples"))

# ---------------------------------------------------------------------------
# Imports from sibling projects
# ---------------------------------------------------------------------------
from src.model_pipeline import BaselineDetector
from harness.budget import PerturbationBudget
from harness.content_attacks import DynamicSynonymSubstitution

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
NUM_SAMPLES = 10
BUDGET_STEPS = 10                       # 0.01, 0.02, ..., 1.00
BUDGETS = [round(i / BUDGET_STEPS, 2) for i in range(1, BUDGET_STEPS + 1)]

DATA_PATH = PROJECT_ROOT / "texts.json"
MODEL_PATH = PROJECT_ROOT / "detector_zoo" / "models" / "logistic_regression"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_malicious_samples(path: Path, n: int, seed: int) -> list[dict]:
    """Load texts.json, keep only label==1, randomly sample n items."""
    print(f"Loading data from {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    malicious = [item for item in data if item.get("label") == 1]
    print(f"  Total entries: {len(data)}, malicious: {len(malicious)}")

    if len(malicious) < n:
        print(f"  WARNING: only {len(malicious)} malicious samples available (requested {n})")
        n = len(malicious)

    random.seed(seed)
    samples = random.sample(malicious, n)
    print(f"  Sampled {len(samples)} malicious texts (seed={seed})")
    return samples


def load_detector(model_path: Path) -> BaselineDetector:
    """Load the pre-trained baseline detector."""
    print(f"Loading detector from {model_path} ...")
    model = BaselineDetector.load_model(str(model_path))
    print("  Detector loaded successfully.")
    return model


def attack_text(text: str, budget_frac: float, attacker: DynamicSynonymSubstitution) -> str:
    """Apply the adversarial transformation to a single text at a given budget."""
    word_count = len(text.split())
    max_edits = max(1, int(word_count * budget_frac)) if budget_frac > 0 else 0
    budget = PerturbationBudget(max_edits=max_edits)

    try:
        perturbed = attacker.transform(text, budget)
    except Exception:
        perturbed = text  # fallback to original on any failure

    return perturbed


def run_analysis():
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    samples = load_malicious_samples(DATA_PATH, NUM_SAMPLES, SEED)
    detector = load_detector(MODEL_PATH)
    attacker = DynamicSynonymSubstitution()

    # Verify all samples are actually detected as malicious (label 1)
    original_texts = [s["text"] for s in samples]
    original_preds = detector.predict(original_texts)
    original_probas = detector.predict_proba(original_texts)

    detected_count = sum(1 for p in original_preds if p == 1)
    print(f"\nBaseline: {detected_count}/{len(original_texts)} samples detected as malicious before attack.")

    # Only keep samples the detector actually flags as malicious
    valid_indices = [i for i, p in enumerate(original_preds) if p == 1]
    if len(valid_indices) < len(original_texts):
        print(f"  Filtering to {len(valid_indices)} samples that the detector flags as malicious.")
        original_texts = [original_texts[i] for i in valid_indices]
        original_probas = [original_probas[i] for i in valid_indices]

    num_valid = len(original_texts)
    if num_valid == 0:
        print("ERROR: No samples are detected as malicious. Cannot run analysis.")
        sys.exit(1)

    # Original confidence for malicious class (index 1)
    orig_confidences = [prob[1] for prob in original_probas]

    # ------------------------------------------------------------------
    # CSV: raw per-sample results
    # ------------------------------------------------------------------
    raw_csv_path = RESULTS_DIR / "raw_results.csv"
    metrics_csv_path = RESULTS_DIR / "degradation_metrics.csv"

    raw_fieldnames = [
        "budget", "sample_idx", "original_pred", "perturbed_pred",
        "original_confidence", "perturbed_confidence", "evaded"
    ]

    metrics_fieldnames = [
        "budget", "detection_rate", "evasion_rate",
        "mean_orig_confidence", "mean_pert_confidence", "mean_confidence_drop"
    ]

    raw_file = open(raw_csv_path, "w", newline="", encoding="utf-8")
    raw_writer = csv.DictWriter(raw_file, fieldnames=raw_fieldnames)
    raw_writer.writeheader()

    aggregated_rows = []

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------
    total_steps = len(BUDGETS)
    print(f"\nRunning {total_steps} budget levels × {num_valid} samples = {total_steps * num_valid} attack runs ...\n")

    for step_idx, budget_frac in enumerate(BUDGETS):
        pct = int(budget_frac * 100)
        print(f"  [{step_idx + 1:3d}/{total_steps}] Budget = {pct:3d}% ...", end="", flush=True)

        perturbed_texts = []
        for text in original_texts:
            perturbed = attack_text(text, budget_frac, attacker)
            perturbed_texts.append(perturbed)

        # Batch predict
        pert_preds = detector.predict(perturbed_texts)
        pert_probas = detector.predict_proba(perturbed_texts)
        pert_confidences = [prob[1] for prob in pert_probas]

        evasions = 0
        for i in range(num_valid):
            evaded = int(pert_preds[i] != 1)  # flipped away from malicious
            evasions += evaded

            raw_writer.writerow({
                "budget": budget_frac,
                "sample_idx": i,
                "original_pred": 1,
                "perturbed_pred": int(pert_preds[i]),
                "original_confidence": round(orig_confidences[i], 6),
                "perturbed_confidence": round(pert_confidences[i], 6),
                "evaded": evaded,
            })

        evasion_rate = evasions / num_valid
        detection_rate = 1.0 - evasion_rate
        mean_orig_conf = sum(orig_confidences) / num_valid
        mean_pert_conf = sum(pert_confidences) / num_valid
        mean_drop = mean_orig_conf - mean_pert_conf

        aggregated_rows.append({
            "budget": budget_frac,
            "detection_rate": round(detection_rate, 4),
            "evasion_rate": round(evasion_rate, 4),
            "mean_orig_confidence": round(mean_orig_conf, 6),
            "mean_pert_confidence": round(mean_pert_conf, 6),
            "mean_confidence_drop": round(mean_drop, 6),
        })

        print(f"  detection={detection_rate:.2%}  evasion={evasion_rate:.2%}  Δconf={mean_drop:+.4f}")

    raw_file.close()
    print(f"\n  Raw results saved to {raw_csv_path}")

    # ------------------------------------------------------------------
    # CSV: aggregated metrics
    # ------------------------------------------------------------------
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metrics_fieldnames)
        writer.writeheader()
        writer.writerows(aggregated_rows)
    print(f"  Aggregated metrics saved to {metrics_csv_path}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    budgets_pct = [row["budget"] * 100 for row in aggregated_rows]
    det_rates = [row["detection_rate"] * 100 for row in aggregated_rows]
    pert_confs = [row["mean_pert_confidence"] * 100 for row in aggregated_rows]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Primary: detection rate
    color_det = "#e74c3c"
    ax1.set_xlabel("Perturbation Budget (%)", fontsize=13)
    ax1.set_ylabel("Detection Rate (%)", color=color_det, fontsize=13)
    ax1.plot(budgets_pct, det_rates, color=color_det, linewidth=2.2, label="Detection Rate")
    ax1.fill_between(budgets_pct, det_rates, alpha=0.10, color=color_det)
    ax1.tick_params(axis="y", labelcolor=color_det)
    ax1.set_ylim(-2, 105)
    ax1.set_xlim(0, 101)

    # Secondary: mean confidence
    ax2 = ax1.twinx()
    color_conf = "#3498db"
    ax2.set_ylabel("Mean Malicious Confidence (%)", color=color_conf, fontsize=13)
    ax2.plot(budgets_pct, pert_confs, color=color_conf, linewidth=2.2, linestyle="--", label="Mean Confidence")
    ax2.fill_between(budgets_pct, pert_confs, alpha=0.08, color=color_conf)
    ax2.tick_params(axis="y", labelcolor=color_conf)
    ax2.set_ylim(-2, 105)

    # Title & legend
    fig.suptitle("Detector Degradation Under Adversarial Perturbation", fontsize=15, fontweight="bold")
    ax1.set_title("DynamicSynonymSubstitution vs. LogisticRegression (TF-IDF)", fontsize=11, style="italic", pad=10)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=11)

    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = RESULTS_DIR / "degradation_curve.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    print(f"  Degradation curve saved to {plot_path}")

    print("\nDone.")


if __name__ == "__main__":
    run_analysis()
