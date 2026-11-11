#!/usr/bin/env python3
"""
Degradation Curve Analysis
===========================
Measures how a detector's accuracy degrades as the adversarial
perturbation budget increases up to the defined MAX_BUDGET.

Supports running single or multi-attacker evaluations across any detector
from detector_zoo (Logistic Regression, CharCNN, Transformer / DistilBERT)
and adversarial attack harnesses.
"""

import sys
import os
import json
import random
import csv
import argparse
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
# Sibling project imports
# ---------------------------------------------------------------------------
from src.model_pipeline import BaselineDetector
from src.char_cnn import CharCNNDetector
from src.transformer_detector import TransformerDetector

from harness.budget import PerturbationBudget
from harness.attacks import (
    # Content
    TextFoolerAttack,
    # Structure
    HTMLCommentInjection,
    ZeroWidthInjectionAttack,
    HTMLTagInjectionAttack,
    HTMLEntityEncodingAttack,
    DOMWrapperAttack,
    HomoglyphSubstitutionAttack,
    # Metadata
    TimestampSpoofing,
    SenderHeaderAttack,
    AuthenticationResultsAttack,
    ReceivedRoutingAttack,
    XHeaderInjectionAttack,
    SubjectMetadataAttack,
)

# ---------------------------------------------------------------------------
# Registries: Detectors & Attackers
# ---------------------------------------------------------------------------
DETECTOR_REGISTRY = {
    "logistic_regression": {
        "class": BaselineDetector,
        "default_model": PROJECT_ROOT / "detector_zoo" / "models" / "logistic_regression",
        "display_name": "Logistic Regression (TF-IDF)",
        "aliases": ["baseline", "lr", "logistic"],
    },
    "char_cnn": {
        "class": CharCNNDetector,
        "default_model": PROJECT_ROOT / "detector_zoo" / "models" / "char_cnn",
        "display_name": "CharCNN",
        "aliases": ["charcnn", "cnn"],
    },
    "transformer": {
        "class": TransformerDetector,
        "default_model": PROJECT_ROOT / "detector_zoo" / "models" / "transformer",
        "display_name": "Transformer (DistilBERT)",
        "aliases": ["distilbert", "trans"],
    },
}

ATTACKER_REGISTRY = {
    # ── 5 Metadata Attacks ──
    "sender_header": {
        "factory": lambda oracle: SenderHeaderAttack(target_model_oracle=oracle),
        "display_name": "Sender Header Attack",
        "aliases": ["sender", "from_header", "sender_header_attack"],
        "category": "metadata",
    },
    "auth_results": {
        "factory": lambda oracle: AuthenticationResultsAttack(target_model_oracle=oracle),
        "display_name": "Authentication Results Attack",
        "aliases": ["auth", "spf_dkim", "authentication_results", "authentication_results_attack"],
        "category": "metadata",
    },
    "received_routing": {
        "factory": lambda oracle: ReceivedRoutingAttack(target_model_oracle=oracle),
        "display_name": "Received Routing Attack",
        "aliases": ["routing", "received", "mta", "received_routing_attack"],
        "category": "metadata",
    },
    "x_header": {
        "factory": lambda oracle: XHeaderInjectionAttack(target_model_oracle=oracle),
        "display_name": "X-Header Injection Attack",
        "aliases": ["x_headers", "spam_status", "x_header_injection", "x_header_injection_attack"],
        "category": "metadata",
    },
    "subject_metadata": {
        "factory": lambda oracle: SubjectMetadataAttack(target_model_oracle=oracle),
        "display_name": "Subject Metadata Attack",
        "aliases": ["subject", "subject_tag", "subject_metadata_attack"],
        "category": "metadata",
    },

    # ── 5 Structure Attacks ──
    "zero_width": {
        "factory": lambda oracle: ZeroWidthInjectionAttack(target_model_oracle=oracle),
        "display_name": "Zero-Width Injection Attack",
        "aliases": ["zw", "zero_width_space", "zero_width_injection", "zero_width_injection_attack"],
        "category": "structure",
    },
    "html_tag": {
        "factory": lambda oracle: HTMLTagInjectionAttack(target_model_oracle=oracle),
        "display_name": "HTML Tag Injection Attack",
        "aliases": ["tag_injection", "inline_tag", "html_tag_injection", "html_tag_injection_attack"],
        "category": "structure",
    },
    "html_entity": {
        "factory": lambda oracle: HTMLEntityEncodingAttack(target_model_oracle=oracle),
        "display_name": "HTML Entity Encoding Attack",
        "aliases": ["entities", "entity_encoding", "html_entity_encoding", "html_entity_encoding_attack"],
        "category": "structure",
    },
    "dom_wrapper": {
        "factory": lambda oracle: DOMWrapperAttack(target_model_oracle=oracle),
        "display_name": "DOM Wrapper Attack",
        "aliases": ["wrapper", "dom", "dom_structure", "dom_wrapper_attack"],
        "category": "structure",
    },
    "homoglyph": {
        "factory": lambda oracle: HomoglyphSubstitutionAttack(target_model_oracle=oracle),
        "display_name": "Homoglyph Substitution Attack",
        "aliases": ["homoglyphs", "confusables", "homoglyph_substitution", "homoglyph_substitution_attack"],
        "category": "structure",
    },

    # ── Legacy Baseline Attacks ──
    "textfooler": {
        "factory": lambda oracle: TextFoolerAttack(target_model_oracle=oracle),
        "display_name": "TextFooler",
        "aliases": ["text_fooler", "content"],
        "category": "content",
    },
    "html_comment": {
        "factory": lambda oracle=None: HTMLCommentInjection(),
        "display_name": "HTML Comment Injection (Legacy)",
        "aliases": ["comment"],
        "category": "structure",
    },
    "timestamp": {
        "factory": lambda oracle=None: TimestampSpoofing(),
        "display_name": "Timestamp Spoofing (Legacy)",
        "aliases": ["spoofing"],
        "category": "metadata",
    },
}

NEW_ATTACKERS = [
    "sender_header",
    "auth_results",
    "received_routing",
    "x_header",
    "subject_metadata",
    "zero_width",
    "html_tag",
    "html_entity",
    "dom_wrapper",
    "homoglyph",
]


def resolve_detector(name_or_alias: str):
    """Normalize and resolve detector specification from name or alias."""
    clean = name_or_alias.strip().lower().replace("-", "_")
    for key, spec in DETECTOR_REGISTRY.items():
        if clean == key or clean in spec["aliases"]:
            return key, spec
    valid_options = list(DETECTOR_REGISTRY.keys())
    raise ValueError(
        f"Unknown detector '{name_or_alias}'. Choose from: {valid_options}"
    )


def resolve_attacker(name_or_alias: str):
    """Normalize and resolve a single attacker specification from name or alias."""
    clean = name_or_alias.strip().lower().replace("-", "_")
    for key, spec in ATTACKER_REGISTRY.items():
        if clean == key or clean in spec["aliases"]:
            return key, spec
    valid_options = list(ATTACKER_REGISTRY.keys())
    raise ValueError(
        f"Unknown attacker '{name_or_alias}'. Choose from: {valid_options}"
    )


def resolve_attackers(spec_str: str) -> list[str]:
    """
    Resolves an attacker specification string into a list of canonical attacker keys.
    Supports:
    - 'new' or 'new_attacks': the 10 new metadata + structure attacks
    - 'all': all registered attackers
    - 'metadata': all metadata attackers
    - 'structure': all structure attackers
    - Comma-separated list: e.g. 'sender_header,zero_width'
    - Single attacker name or alias
    """
    clean = spec_str.strip().lower()
    if clean in ("new", "new_attacks", "new_attack"):
        return list(NEW_ATTACKERS)
    if clean == "all":
        return list(ATTACKER_REGISTRY.keys())
    if clean in ("metadata", "meta"):
        return [k for k, v in ATTACKER_REGISTRY.items() if v.get("category") == "metadata"]
    if clean in ("structure", "struct"):
        return [k for k, v in ATTACKER_REGISTRY.items() if v.get("category") == "structure"]

    parts = [p.strip() for p in spec_str.split(",") if p.strip()]
    resolved_keys = []
    for part in parts:
        key, _ = resolve_attacker(part)
        if key not in resolved_keys:
            resolved_keys.append(key)
    return resolved_keys


# ---------------------------------------------------------------------------
# Data & Model Helpers
# ---------------------------------------------------------------------------
def load_malicious_samples(path: Path, n: int, seed: int, detector=None) -> list[dict]:
    """
    Loads malicious samples (label=1) deterministically.
    If detector is provided, filters for samples that the detector actually detects (pred=1).
    """
    print(f"Loading data from {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    malicious = [item for item in data if item.get("label") == 1]
    print(f"  Total entries: {len(data)}, malicious: {len(malicious)}")

    random.seed(seed)
    shuffled = random.sample(malicious, len(malicious))

    if detector is not None:
        detected = []
        for s in shuffled:
            pred = detector.predict([s["text"]])[0]
            if pred == 1:
                detected.append(s)
                if len(detected) == n:
                    break
        print(f"  Found {len(detected)} samples detected as malicious by target detector.")
        return detected

    if len(malicious) < n:
        print(f"  WARNING: only {len(malicious)} malicious samples available")
        n = len(malicious)
    return shuffled[:n]


def load_detector(detector_type: str, model_path: Path | str | None = None):
    key, spec = resolve_detector(detector_type)
    resolved_path = Path(model_path) if model_path else spec["default_model"]
    print(f"Loading detector [{spec['display_name']}] from {resolved_path} ...")
    if not resolved_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {resolved_path}")
    detector = spec["class"].load_model(str(resolved_path))
    return detector, key, spec, resolved_path


def create_oracle(detector):
    """Creates a black-box oracle callable for the target detector."""
    def oracle(text: str):
        pred = detector.predict([text])[0]
        probs = detector.predict_proba([text])[0]
        label = "phishing" if pred == 1 else "benign"
        return (label, float(probs[pred]))
    return oracle


def attack_text(text: str, budget_frac: float, attacker) -> tuple[str, int, bool]:
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

    # Calculate real budget consumed (word differences or budget.consumed)
    orig_words = text.split()
    pert_words = perturbed.split()
    min_len = min(len(orig_words), len(pert_words))

    consumed = sum(1 for i in range(min_len) if orig_words[i] != pert_words[i])
    consumed += abs(len(orig_words) - len(pert_words))
    if consumed == 0 and budget.consumed > 0:
        consumed = budget.consumed

    # If budget was requested but no edits were made, mark as failed
    if budget_frac > 0 and consumed == 0 and perturbed == text:
        success = False

    return perturbed, consumed, success


# ---------------------------------------------------------------------------
# Core Analysis Pipeline
# ---------------------------------------------------------------------------
def run_single_analysis(
    detector_type: str = "logistic_regression",
    model_path: Path | str | None = None,
    attacker_type: str = "sender_header",
    num_samples: int = 10,
    max_budget: float = 0.20,
    budget_steps: int = 3,
    seed: int = 42,
    data_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    timestamp: str | None = None,
    skip_flipped: bool = False,
):
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    data_file = Path(data_path) if data_path else PROJECT_ROOT / "datasets" / "texts.json"
    detector, det_key, det_spec, resolved_model_path = load_detector(detector_type, model_path)
    att_key, att_spec = resolve_attacker(attacker_type)

    if output_dir:
        results_dir = Path(output_dir)
    else:
        results_dir = Path(__file__).resolve().parent / "results" / timestamp / f"{det_key}_{att_key}"

    results_dir.mkdir(parents=True, exist_ok=True)

    # Calculate budget sweep
    budgets = [round((i / budget_steps) * max_budget, 4) for i in range(1, budget_steps + 1)]

    # Load dataset (filtered for samples detected as malicious)
    samples = load_malicious_samples(data_file, num_samples, seed, detector=detector)

    # Initialize oracle and attacker
    oracle = create_oracle(detector)
    attacker = att_spec["factory"](oracle)

    original_texts = [s["text"] for s in samples]
    original_preds = detector.predict(original_texts)
    original_probas = detector.predict_proba(original_texts)

    valid_indices = [i for i, p in enumerate(original_preds) if p == 1]
    if len(valid_indices) < len(original_texts):
        print(f"  Note: {len(original_texts) - len(valid_indices)}/{len(original_texts)} samples "
              f"were not initially detected as malicious; evaluating on {len(valid_indices)} detected samples.")
        original_texts = [original_texts[i] for i in valid_indices]
        original_probas = [original_probas[i] for i in valid_indices]

    num_valid = len(original_texts)
    if num_valid == 0:
        print("ERROR: No samples are detected as malicious by this detector. Cannot run analysis.")
        return None

    orig_confidences = [prob[1] for prob in original_probas]

    # ------------------------------------------------------------------
    # CSV Configuration
    # ------------------------------------------------------------------
    raw_csv_path = results_dir / "raw_results.csv"
    metrics_csv_path = results_dir / "degradation_metrics.csv"
    config_json_path = results_dir / "run_config.json"

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
    total_steps = len(budgets)
    print(f"\nTarget Detector : {det_spec['display_name']} ({resolved_model_path.name})")
    print(f"Attacker        : {att_spec['display_name']}")
    print(f"Saving results to: {results_dir}")
    print(f"Running {total_steps} budget levels × {num_valid} samples = {total_steps * num_valid} attack runs ...\n")

    # Track samples that have already flipped detection (pred != 1)
    flipped_samples = {}  # sample_idx -> dict of cached results

    for step_idx, budget_frac in enumerate(budgets):
        pct = int(budget_frac * 100)

        # Determine which samples need attacking in this budget step
        if skip_flipped:
            active_indices = [i for i in range(num_valid) if i not in flipped_samples]
            skipped_count = num_valid - len(active_indices)
        else:
            active_indices = list(range(num_valid))
            skipped_count = 0

        status_prefix = f"  [{step_idx + 1:3d}/{total_steps}] Budget = {pct:3d}%"
        if skip_flipped and skipped_count > 0:
            print(f"{status_prefix} (active: {len(active_indices)}/{num_valid}, skipped: {skipped_count}) ...", end="", flush=True)
        else:
            print(f"{status_prefix} ...", end="", flush=True)

        active_perturbed = {}
        active_consumed = {}
        active_success = {}

        for i in active_indices:
            text = original_texts[i]
            pert, cons, succ = attack_text(text, budget_frac, attacker)
            active_perturbed[i] = pert
            active_consumed[i] = cons
            active_success[i] = succ

        if active_indices:
            active_texts = [active_perturbed[i] for i in active_indices]
            active_preds = detector.predict(active_texts)
            active_probas = detector.predict_proba(active_texts)
            active_confidences = [prob[1] for prob in active_probas]
        else:
            active_preds = []
            active_confidences = []

        # Build full result lists for all samples in this step
        step_perturbed_texts = []
        step_consumed_budgets = []
        step_success_flags = []
        step_pert_preds = []
        step_pert_confidences = []

        active_pos = 0
        for i in range(num_valid):
            if skip_flipped and i in flipped_samples:
                prev = flipped_samples[i]
                step_perturbed_texts.append(prev["perturbed_text"])
                step_consumed_budgets.append(prev["consumed_budget"])
                step_success_flags.append(prev["success"])
                step_pert_preds.append(prev["perturbed_pred"])
                step_pert_confidences.append(prev["perturbed_confidence"])
            else:
                p_text = active_perturbed[i]
                p_cons = active_consumed[i]
                p_succ = active_success[i]
                p_pred = int(active_preds[active_pos])
                p_conf = float(active_confidences[active_pos])
                active_pos += 1

                step_perturbed_texts.append(p_text)
                step_consumed_budgets.append(p_cons)
                step_success_flags.append(p_succ)
                step_pert_preds.append(p_pred)
                step_pert_confidences.append(p_conf)

                if skip_flipped and p_pred != 1:
                    flipped_samples[i] = {
                        "perturbed_text": p_text,
                        "consumed_budget": p_cons,
                        "success": p_succ,
                        "perturbed_pred": p_pred,
                        "perturbed_confidence": p_conf,
                    }

        evasions = 0
        for i in range(num_valid):
            evaded = int(step_pert_preds[i] != 1)
            evasions += evaded

            orig_b64 = base64.b64encode(original_texts[i].encode("utf-8")).decode("utf-8")
            pert_b64 = base64.b64encode(step_perturbed_texts[i].encode("utf-8")).decode("utf-8")

            raw_writer.writerow({
                "budget": budget_frac,
                "sample_idx": i,
                "original_pred": 1,
                "perturbed_pred": int(step_pert_preds[i]),
                "original_confidence": round(orig_confidences[i], 6),
                "perturbed_confidence": round(step_pert_confidences[i], 6),
                "evaded": evaded,
                "real_budget_consumed": step_consumed_budgets[i],
                "transformation_success": step_success_flags[i],
                "original_text_b64": orig_b64,
                "perturbed_text_b64": pert_b64
            })

        evasion_rate = evasions / num_valid
        detection_rate = 1.0 - evasion_rate

        se_det = (detection_rate * (1.0 - detection_rate) / num_valid) ** 0.5 if num_valid > 0 else 0.0

        mean_orig_conf = sum(orig_confidences) / num_valid
        mean_pert_conf = sum(step_pert_confidences) / num_valid
        std_pert_conf = statistics.stdev(step_pert_confidences) if num_valid > 1 else 0.0
        mean_drop = mean_orig_conf - mean_pert_conf

        mean_consumed = sum(step_consumed_budgets) / num_valid
        failed_count = sum(1 for s in step_success_flags if not s)

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

        skip_info = f"  skipped={skipped_count}" if skip_flipped else ""
        print(f"  det={detection_rate:.2%}  Δconf={mean_drop:+.4f}  failures={failed_count}{skip_info}")

    raw_file.close()

    # ------------------------------------------------------------------
    # Metrics CSV Output
    # ------------------------------------------------------------------
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metrics_fieldnames)
        writer.writeheader()
        writer.writerows(aggregated_rows)

    # ------------------------------------------------------------------
    # Save Metadata & Run Configuration
    # ------------------------------------------------------------------
    run_config = {
        "timestamp": timestamp,
        "detector_key": f"{det_key}_{att_key}",
        "detector_display_name": f"{det_spec['display_name']} - {att_spec['display_name']}",
        "model_path": str(resolved_model_path),
        "attacker_key": att_key,
        "attacker_display_name": att_spec["display_name"],
        "num_samples_requested": num_samples,
        "num_samples_valid": num_valid,
        "seed": seed,
        "max_budget": max_budget,
        "budget_steps": budget_steps,
        "skip_flipped": skip_flipped,
        "data_path": str(data_file),
    }
    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

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
    ax1.set_xlim(0, (max_budget * 100) + 1)

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

    # Dynamic Title & Legend
    fig.suptitle("Detector Degradation Under Adversarial Perturbation", fontsize=15, fontweight="bold")
    ax1.set_title(f"{att_spec['display_name']} vs. {det_spec['display_name']}", fontsize=11, style="italic", pad=10)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=11)

    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = results_dir / "degradation_curve.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    print(f"\nDone. Results saved in {results_dir}")
    return results_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Degradation Curve Analysis: evaluate detector robustness under adversarial perturbation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    all_detectors = list(DETECTOR_REGISTRY.keys()) + ["all"]
    all_aliases = [alias for s in DETECTOR_REGISTRY.values() for alias in s["aliases"]]
    parser.add_argument(
        "--detector", "-d",
        type=str,
        default="logistic_regression",
        choices=all_detectors + all_aliases,
        help="Detector to evaluate ('logistic_regression', 'char_cnn', 'transformer', or 'all')."
    )
    parser.add_argument(
        "--model-path", "-m",
        type=str,
        default=None,
        help="Custom path to trained model (overrides default path for the chosen detector)."
    )
    parser.add_argument(
        "--attacker", "-a",
        type=str,
        default="new",
        help="Adversarial attack algorithm to apply. Can be 'new' (runs all 10 new attacks), 'all', 'metadata', 'structure', a specific attack name, or comma-separated attack names."
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=10,
        help="Number of malicious samples to evaluate."
    )
    parser.add_argument(
        "--max-budget", "-b",
        type=float,
        default=0.20,
        help="Maximum perturbation budget fraction (0.20 = 20%%)."
    )
    parser.add_argument(
        "--budget-steps", "-s",
        type=int,
        default=3,
        help="Number of evenly spaced budget steps from >0 up to max-budget."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Custom path to texts.json dataset."
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Custom output directory to store results."
    )
    parser.add_argument(
        "--report", "--visualize",
        action="store_true",
        default=True,
        help="Automatically generate an interactive HTML report after analysis completes."
    )
    parser.add_argument(
        "--skip-flipped", "--skip-if-flipped", "--early-stop",
        dest="skip_flipped",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip running attack transformations on samples that already flipped detection in a previous budget step, speeding up degradation curve generation."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Resolve detectors
    if args.detector.lower() == "all":
        detectors_to_run = list(DETECTOR_REGISTRY.keys())
    else:
        det_key, _ = resolve_detector(args.detector)
        detectors_to_run = [det_key]

    # Resolve attackers
    attackers_to_run = resolve_attackers(args.attacker)

    total_runs = len(detectors_to_run) * len(attackers_to_run)
    print(f"Executing {total_runs} separate analysis run(s) [{len(detectors_to_run)} detector(s) × {len(attackers_to_run)} attacker(s)]")
    print(f"Attackers: {', '.join(attackers_to_run)}")
    print(f"Samples: {args.samples} | Budget Steps: {args.budget_steps} | Max Budget: {args.max_budget:.2%} | Skip Flipped: {args.skip_flipped}")

    run_idx = 0
    base_results_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent / "results" / timestamp

    for det in detectors_to_run:
        for att in attackers_to_run:
            run_idx += 1
            print(f"\n{'='*75}")
            print(f"Run [{run_idx}/{total_runs}]: Detector={det} | Attacker={att}")
            print(f"{'='*75}")

            sub_output_dir = base_results_dir / f"{det}_{att}" if total_runs > 1 else base_results_dir

            run_single_analysis(
                detector_type=det,
                model_path=args.model_path,
                attacker_type=att,
                num_samples=args.samples,
                max_budget=args.max_budget,
                budget_steps=args.budget_steps,
                seed=args.seed,
                data_path=args.data_path,
                output_dir=sub_output_dir,
                timestamp=timestamp,
                skip_flipped=args.skip_flipped,
            )

    if args.report:
        try:
            from visualize_results import resolve_results_directory, load_detector_run, build_html_report
            target_dir = base_results_dir
            base_dir, detector_dirs = resolve_results_directory(str(target_dir))
            data = [load_detector_run(d) for d in detector_dirs]
            html_content = build_html_report(f"Degradation Analysis ({timestamp})", base_dir, data)
            report_file = base_dir / "report.html"
            report_file.write_text(html_content, encoding="utf-8")
            print(f"\n[REPORT] Interactive HTML report created: {report_file}")
        except Exception as e:
            import traceback
            print(f"\n[WARNING] Could not auto-generate report: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
