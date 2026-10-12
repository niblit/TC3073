#!/usr/bin/env python3
"""
Degradation Curve Analysis
===========================
Measures how a detector's accuracy degrades as the adversarial
perturbation budget increases up to the defined MAX_BUDGET.
"""

import sys
import os
import json
import random
import csv
import warnings
import logging
import base64
import statistics
from datetime import datetime
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
MAX_BUDGET = 0.20  # Maximum perturbation budget (20%)
BUDGET_STEPS = 20  # Number of steps to reach MAX_BUDGET

# Calculates evenly spaced budgets from >0 up to MAX_BUDGET
BUDGETS = [round((i / BUDGET_STEPS) * MAX_BUDGET, 4) for i in range(1, BUDGET_STEPS + 1)]

DATA_PATH = PROJECT_ROOT / "datasets" / "texts.json"
MODEL_PATH = PROJECT_ROOT / "detector_zoo" / "models" / "logistic_regression"

# Dynamic output directory: results/YYYYMMDD-HHMMSS/model_name
TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
RESULTS_DIR = Path(__file__).resolve().parent / "results" / TIMESTAMP / MODEL_PATH.name


def load_malicious_samples(path: Path, n: int, seed: int) -> list[dict]:
    print(f"Loading data from {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    malicious = [item for item in data if item.get("label") == 1]
    print(f"  Total entries: {len(data)}, malicious: {len(malicious)}")

    if len(malicious) < n:
        print(f"  WARNING: only {len(malicious)} malicious samples available")
        n = len(malicious)

    random.seed(seed)
    return random.sample(malicious, n)


def load_detector(model_path: Path) -> BaselineDetector:
    print(f"Loading detector from {model_path} ...")
    return BaselineDetector.load_model(str(model_path))


def attack_text(text: str, budget_frac: float, attacker: DynamicSynonymSubstitution) -> tuple[str, int, bool]:
    word_count = len(text.split())
    max_edits = max(1, int(word_count * budget_frac)) if budget_frac > 0 else 0
    budget = PerturbationBudget(max_edits=max_edits)

    success = True
    try:
        perturbed = attacker.transform(text, budget)
    except Exception as e:
        import traceback
        print(f"\n[ATTACK ERROR]: {type(e).__name__}: {e}")
        traceback.print_exc()
        perturbed = text
        success = False

    # Calculate real budget consumed (word differences)
    orig_words = text.split()
    pert_words = perturbed.split()
    min_len = min(len(orig_words), len(pert_words))
    
    consumed = sum(1 for i in range(min_len) if orig_words[i] != pert_words[i])
    consumed += abs(len(orig_words) - len(pert_words))

    # If budget was requested but no edits were made, mark as failed
    if budget_frac > 0 and consumed == 0:
        success = False

    return perturbed, consumed, success


def run_analysis():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    samples = load_malicious_samples(DATA_PATH, NUM_SAMPLES, SEED)
    detector = load_detector(MODEL_PATH)

    def oracle(text: str):
        pred = detector.predict([text])[0]
        probs = detector.predict_proba([text])[0]
        label = "phishing" if pred == 1 else "benign"
        return (label, float(probs[pred]))

    attacker = DynamicSynonymSubstitution(target_model_oracle=oracle)

    original_texts = [s["text"] for s in samples]
    original_preds = detector.predict(original_texts)
    original_probas = detector.predict_proba(original_texts)

    valid_indices = [i for i, p in enumerate(original_preds) if p == 1]
    if len(valid_indices) < len(original_texts):
        original_texts = [original_texts[i] for i in valid_indices]
        original_probas = [original_probas[i] for i in valid_indices]

    num_valid = len(original_texts)
    if num_valid == 0:
        print("ERROR: No samples are detected as malicious. Cannot run analysis.")
        sys.exit(1)

    orig_confidences = [prob[1] for prob in original_probas]

    # ------------------------------------------------------------------
    # CSV Configuration
    # ------------------------------------------------------------------
    raw_csv_path = RESULTS_DIR / "raw_results.csv"
    metrics_csv_path = RESULTS_DIR / "degradation_metrics.csv"

    raw_fieldnames = [
        "budget", "sample_idx", "original_pred", "perturbed_pred",
        "original_confidence", "perturbed_confidence", "evaded",
        "real_budget_consumed", "transformation_success",
        "original_text_b64", "perturbed_text_b64"
    ]

    metrics_fieldnames = [
        "budget", "detection_rate", "se_detection_rate", "evasion_rate",
        "mean_orig_confidence", "mean_pert_confidence", "std_pert_confidence", 
        "mean_confidence_drop", "mean_real_budget_consumed", "failed_transformations"
    ]

    raw_file = open(raw_csv_path, "w", newline="", encoding="utf-8")
    raw_writer = csv.DictWriter(raw_file, fieldnames=raw_fieldnames)
    raw_writer.writeheader()

    aggregated_rows = []

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------
    total_steps = len(BUDGETS)
    print(f"\nSaving results to: {RESULTS_DIR}")
    print(f"Running {total_steps} budget levels × {num_valid} samples = {total_steps * num_valid} attack runs ...\n")

    for step_idx, budget_frac in enumerate(BUDGETS):
        pct = int(budget_frac * 100)
        print(f"  [{step_idx + 1:3d}/{total_steps}] Budget = {pct:3d}% ...", end="", flush=True)

        perturbed_texts = []
        consumed_budgets = []
        success_flags = []
        
        for text in original_texts:
            pert, cons, succ = attack_text(text, budget_frac, attacker)
            perturbed_texts.append(pert)
            consumed_budgets.append(cons)
            success_flags.append(succ)

        pert_preds = detector.predict(perturbed_texts)
        pert_probas = detector.predict_proba(perturbed_texts)
        pert_confidences = [prob[1] for prob in pert_probas]

        evasions = 0
        for i in range(num_valid):
            evaded = int(pert_preds[i] != 1)
            evasions += evaded

            orig_b64 = base64.b64encode(original_texts[i].encode("utf-8")).decode("utf-8")
            pert_b64 = base64.b64encode(perturbed_texts[i].encode("utf-8")).decode("utf-8")

            raw_writer.writerow({
                "budget": budget_frac,
                "sample_idx": i,
                "original_pred": 1,
                "perturbed_pred": int(pert_preds[i]),
                "original_confidence": round(orig_confidences[i], 6),
                "perturbed_confidence": round(pert_confidences[i], 6),
                "evaded": evaded,
                "real_budget_consumed": consumed_budgets[i],
                "transformation_success": success_flags[i],
                "original_text_b64": orig_b64,
                "perturbed_text_b64": pert_b64
            })

        evasion_rate = evasions / num_valid
        detection_rate = 1.0 - evasion_rate

        # Calculate standard error of the sample proportion
        se_det = (detection_rate * (1.0 - detection_rate) / num_valid) ** 0.5 if num_valid > 0 else 0.0



        mean_orig_conf = sum(orig_confidences) / num_valid
        mean_pert_conf = sum(pert_confidences) / num_valid
        std_pert_conf = statistics.stdev(pert_confidences) if num_valid > 1 else 0.0
        mean_drop = mean_orig_conf - mean_pert_conf
        
        mean_consumed = sum(consumed_budgets) / num_valid
        failed_count = sum(1 for s in success_flags if not s)

        aggregated_rows.append({
            "budget": budget_frac,
            "detection_rate": round(detection_rate, 4),
            "se_detection_rate": round(se_det, 4),
            "evasion_rate": round(evasion_rate, 4),
            "mean_orig_confidence": round(mean_orig_conf, 6),
            "mean_pert_confidence": round(mean_pert_conf, 6),
            "std_pert_confidence": round(std_pert_conf, 6),
            "mean_confidence_drop": round(mean_drop, 6),
            "mean_real_budget_consumed": round(mean_consumed, 2),
            "failed_transformations": failed_count
        })

        print(f"  det={detection_rate:.2%}  Δconf={mean_drop:+.4f}  failures={failed_count}")

    raw_file.close()

    # ------------------------------------------------------------------
    # Metrics CSV Output
    # ------------------------------------------------------------------
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metrics_fieldnames)
        writer.writeheader()
        writer.writerows(aggregated_rows)

    # ------------------------------------------------------------------
    # Plotting with Variance
    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    budgets_pct = [row["budget"] * 100 for row in aggregated_rows]
    det_rates = [row["detection_rate"] * 100 for row in aggregated_rows]
    det_se = [row["se_detection_rate"] * 100 for row in aggregated_rows]
    pert_confs = [row["mean_pert_confidence"] * 100 for row in aggregated_rows]
    stdevs = [row["std_pert_confidence"] * 100 for row in aggregated_rows]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Primary: detection rate
    color_det = "#e74c3c"
    ax1.set_xlabel("Perturbation Budget (%)", fontsize=13)
    ax1.set_ylabel("Detection Rate (%)", color=color_det, fontsize=13)
    ax1.plot(budgets_pct, det_rates, color=color_det, linewidth=2.2, label="Detection Rate")

    # Variance band
    det_upper = [min(100, d + s) for d, s in zip(det_rates, det_se)]
    det_lower = [max(0, d - s) for d, s in zip(det_rates, det_se)]
    ax1.fill_between(budgets_pct, det_lower, det_upper, alpha=0.15, color=color_det, label="Detection Rate ±1 SE")

    ax1.tick_params(axis="y", labelcolor=color_det)
    ax1.set_ylim(-2, 105)
    ax1.set_xlim(0, (MAX_BUDGET * 100) + 1)

    # Secondary: mean confidence with standard deviation variance bands
    ax2 = ax1.twinx()
    color_conf = "#3498db"
    ax2.set_ylabel("Mean Malicious Confidence (%)", color=color_conf, fontsize=13)
    
    upper_bound = [min(100, c + s) for c, s in zip(pert_confs, stdevs)]
    lower_bound = [max(0, c - s) for c, s in zip(pert_confs, stdevs)]
    
    ax2.plot(budgets_pct, pert_confs, color=color_conf, linewidth=2.2, linestyle="--", label="Mean Confidence")
    ax2.fill_between(budgets_pct, lower_bound, upper_bound, alpha=0.15, color=color_conf, label="Confidence ±1σ")
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
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    print(f"\nDone. Results saved in {RESULTS_DIR}")

if __name__ == "__main__":
    run_analysis()
