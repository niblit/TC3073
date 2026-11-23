#!/usr/bin/env python3
"""
Cross-Detector Transferability Analysis
=======================================
Evaluates how adversarial problem-space phishing/BEC examples crafted against
a source detector oracle (O_S) at max perturbation budget (budget = 0.20)
transfer to other target detectors (f_T != O_S) across all 42 benchmark runs
(14 attacks x 3 source detectors, N = 1,000 samples, seed = 3073).

Computes both:
  a) Overall Transfer Evasion Rate & Mean Confidence Drop (Delta Conf) across
     S_valid(O_S) cap S_valid(f_T) (samples initially detected as malicious at
     epsilon = 0 by both source oracle O_S and target detector f_T, making
     off-diagonal rates directly comparable to diagonal direct evasion rates).
  b) Conditional Transfer Success Rate among samples that successfully flipped
     the source oracle (evaded == 1) and belonged to S_valid(f_T), i.e.,
     S_evaded(O_S) cap S_valid(f_T).

Outputs:
  - transfer_matrix.csv
  - transfer_summary.txt
  - transfer_results.json
in /home/omen/Projects/Code/TC3073/UsefulResullts/transferability_1000_samples/
"""

import argparse
import base64
import csv
import io
import json
import logging
import math
import os
import statistics
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Silence noisy libraries BEFORE importing them
# ---------------------------------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Project root & detector_zoo imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "detector_zoo"))

import numpy as np
import torch

from src.model_pipeline import BaselineDetector
from src.char_cnn import CharCNNDetector
from src.transformer_detector import TransformerDetector

# ---------------------------------------------------------------------------
# Canonical Detector & Attack Specifications
# ---------------------------------------------------------------------------
DETECTOR_ORDER = ["logistic_regression", "char_cnn", "transformer"]

DETECTOR_SPECS = {
    "logistic_regression": {
        "class": BaselineDetector,
        "model_path": PROJECT_ROOT / "detector_zoo" / "models" / "logistic_regression",
        "display_name": "Logistic Regression (TF-IDF)",
        "short_name": "LogReg (TF-IDF)",
        "abbr": "LR",
    },
    "char_cnn": {
        "class": CharCNNDetector,
        "model_path": PROJECT_ROOT / "detector_zoo" / "models" / "char_cnn",
        "display_name": "CharCNN",
        "short_name": "CharCNN",
        "abbr": "CharCNN",
    },
    "transformer": {
        "class": TransformerDetector,
        "model_path": PROJECT_ROOT / "detector_zoo" / "models" / "transformer",
        "display_name": "Transformer (DistilBERT)",
        "short_name": "Transformer",
        "abbr": "Trans",
    },
}

SINGLE_ATTACK_KEYS = [
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
    "textfooler",
    "html_comment",
    "timestamp",
]

ALL_ATTACK_KEYS = SINGLE_ATTACK_KEYS + ["combine_all"]

ATTACK_METADATA = {
    "sender_header": {
        "display_name": "Sender Header Attack",
        "category": "metadata",
    },
    "auth_results": {
        "display_name": "Authentication Results Attack",
        "category": "metadata",
    },
    "received_routing": {
        "display_name": "Received Routing Attack",
        "category": "metadata",
    },
    "x_header": {
        "display_name": "X-Header Injection Attack",
        "category": "metadata",
    },
    "subject_metadata": {
        "display_name": "Subject Metadata Attack",
        "category": "metadata",
    },
    "zero_width": {
        "display_name": "Zero-Width Injection Attack",
        "category": "structure",
    },
    "html_tag": {
        "display_name": "HTML Tag Injection Attack",
        "category": "structure",
    },
    "html_entity": {
        "display_name": "HTML Entity Encoding Attack",
        "category": "structure",
    },
    "dom_wrapper": {
        "display_name": "DOM Wrapper Attack",
        "category": "structure",
    },
    "homoglyph": {
        "display_name": "Homoglyph Substitution Attack",
        "category": "structure",
    },
    "textfooler": {
        "display_name": "TextFooler",
        "category": "content",
    },
    "html_comment": {
        "display_name": "HTML Comment Injection (Legacy)",
        "category": "structure",
    },
    "timestamp": {
        "display_name": "Timestamp Spoofing (Legacy)",
        "category": "metadata",
    },
    "combine_all": {
        "display_name": "Combined All Attacks (Best-of-K)",
        "category": "combined",
    },
}

DEFAULT_SINGLE_RUNS_DIR = (
    PROJECT_ROOT / "UsefulResullts" / "samples1000attacks13" / "20261121-070628"
)
DEFAULT_COMBINED_RUNS_DIR = (
    PROJECT_ROOT / "UsefulResullts" / "combined_big_run_1000_samples" / "20261121-013847"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "UsefulResullts" / "transferability_1000_samples"
)


def format_duration(seconds: float) -> str:
    """Format seconds into a human-readable mm:ss or hh:mm:ss string."""
    if seconds < 0 or math.isnan(seconds) or math.isinf(seconds):
        return "--:--"
    total_sec = int(round(seconds))
    mins, secs = divmod(total_sec, 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs:d}h {mins:02d}m {secs:02d}s"
    return f"{mins:02d}m {secs:02d}s"


# ---------------------------------------------------------------------------
# Fast CSV Extraction for a Target Budget Step
# ---------------------------------------------------------------------------
def extract_budget_rows(csv_path: Path, target_budget: float = 0.2) -> dict[int, dict]:
    """
    Fast extraction of rows matching `budget == target_budget` from `raw_results.csv`.
    Pre-filters lines by prefix before invoking csv.DictReader to avoid parsing
    the other 39 budget steps (~97.5% reduction in I/O and base64 parsing time).
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing raw_results.csv: {csv_path}")

    target_prefixes = (
        f"{target_budget},",
        f"{target_budget:.1f},",
        f"{target_budget:.2f},",
        f"{target_budget:.3f},",
        f"{target_budget:.4f},",
    )

    matching_lines = []
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline()
        if not header:
            raise ValueError(f"Empty CSV file: {csv_path}")
        for line in f:
            if line.startswith(target_prefixes):
                matching_lines.append(line)
            else:
                comma_idx = line.find(",")
                if comma_idx > 0:
                    try:
                        if abs(float(line[:comma_idx]) - target_budget) < 1e-6:
                            matching_lines.append(line)
                    except ValueError:
                        pass

    reader = csv.DictReader(io.StringIO(header + "".join(matching_lines)))
    records_by_idx: dict[int, dict] = {}

    for row in reader:
        sample_idx = int(row["sample_idx"])
        orig_b64 = row["original_text_b64"]
        pert_b64 = row["perturbed_text_b64"]
        orig_text = base64.b64decode(orig_b64).decode("utf-8")
        pert_text = base64.b64decode(pert_b64).decode("utf-8")

        evaded_str = str(row["evaded"]).strip().lower()
        evaded = 1 if evaded_str in ("1", "true") else 0

        succ_str = str(row["transformation_success"]).strip().lower()
        trans_success = succ_str in ("1", "true")

        records_by_idx[sample_idx] = {
            "budget": float(row["budget"]),
            "sample_idx": sample_idx,
            "original_pred": int(row["original_pred"]),
            "perturbed_pred": int(row["perturbed_pred"]),
            "original_confidence": float(row["original_confidence"]),
            "perturbed_confidence": float(row["perturbed_confidence"]),
            "evaded": evaded,
            "real_budget_consumed": float(row["real_budget_consumed"]),
            "transformation_success": trans_success,
            "original_text": orig_text,
            "perturbed_text": pert_text,
        }

    return records_by_idx


# ---------------------------------------------------------------------------
# Batched Detector Inference with Caching
# ---------------------------------------------------------------------------
def load_all_detectors(batch_size: int = 128) -> dict[str, object]:
    """Load all 3 trained detectors from detector_zoo/models/."""
    detectors = {}
    print("\n[1/4] Loading Trained Detectors from detector_zoo/models/ ...")
    t_start = time.time()
    for det_key in DETECTOR_ORDER:
        spec = DETECTOR_SPECS[det_key]
        t0 = time.time()
        model_path = spec["model_path"]
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        detector = spec["class"].load_model(str(model_path))
        if hasattr(detector, "batch_size"):
            detector.batch_size = batch_size
        device_info = getattr(detector, "device", "cpu")
        print(
            f"  + Loaded {spec['display_name']:28s} | device={str(device_info):4s} "
            f"| elapsed={time.time() - t0:.2f}s"
        )
        detectors[det_key] = detector
    print(f"  All 3 detectors loaded in {time.time() - t_start:.2f}s.\n")
    return detectors


def batched_predict_malicious_proba_and_label(
    detector_key: str,
    detector: object,
    texts: list[str],
    batch_size: int = 128,
    cache: dict[str, tuple[int, float]] | None = None,
) -> tuple[list[int], list[float], int, int]:
    """
    Evaluates a list of texts on `detector`, returning:
      (predicted_labels, malicious_probabilities_p1, num_cache_hits, num_inferred)
    Uses `cache` (mapping text -> (pred_label, prob_class_1)) to avoid redundant
    forward passes on duplicate or unperturbed strings, and runs a single
    `predict_proba` pass per unique uncached batch (deriving `pred_label` directly
    from the probability vector exactly as each detector's `predict()` method does).
    """
    if not texts:
        return [], [], 0, 0

    if cache is None:
        cache = {}

    missing_unique_texts: list[str] = []
    missing_seen: set[str] = set()
    cache_hits = 0

    for t in texts:
        if t in cache:
            cache_hits += 1
        elif t not in missing_seen:
            missing_seen.add(t)
            missing_unique_texts.append(t)

    num_inferred = len(missing_unique_texts)
    if num_inferred > 0:
        if detector_key == "logistic_regression":
            probs = detector.predict_proba(missing_unique_texts)
            preds = detector.predict(missing_unique_texts)
            for t, pred, prob_row in zip(missing_unique_texts, preds, probs):
                cache[t] = (int(pred), float(prob_row[1]))
        elif detector_key == "char_cnn":
            probs = detector.predict_proba(missing_unique_texts)
            preds = (probs[:, 1] >= 0.5).astype(int)
            for t, pred, prob_row in zip(missing_unique_texts, preds, probs):
                cache[t] = (int(pred), float(prob_row[1]))
        elif detector_key == "transformer":
            # Tokenize & run in batches of `batch_size` to avoid DataLoader row-by-row overhead
            model = detector.model
            if model.training:
                model.eval()
            device = detector.device
            for start_idx in range(0, num_inferred, batch_size):
                chunk = missing_unique_texts[start_idx : start_idx + batch_size]
                enc = detector._tokenize(chunk)
                with torch.no_grad():
                    input_ids = enc["input_ids"].to(device, non_blocking=True)
                    attention_mask = enc["attention_mask"].to(device, non_blocking=True)
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                    probs_chunk = torch.softmax(logits, dim=-1).cpu().numpy()
                preds_chunk = np.argmax(probs_chunk, axis=1)
                for t, pred, prob_row in zip(chunk, preds_chunk, probs_chunk):
                    cache[t] = (int(pred), float(prob_row[1]))
        else:
            probs = detector.predict_proba(missing_unique_texts)
            preds = detector.predict(missing_unique_texts)
            for t, pred, prob_row in zip(missing_unique_texts, preds, probs):
                cache[t] = (int(pred), float(prob_row[1]))

    out_preds = [cache[t][0] for t in texts]
    out_probs = [cache[t][1] for t in texts]
    return out_preds, out_probs, cache_hits, num_inferred


# ---------------------------------------------------------------------------
# Main Transferability Pipeline
# ---------------------------------------------------------------------------
def run_transferability_analysis(
    single_runs_dir: Path = DEFAULT_SINGLE_RUNS_DIR,
    combined_runs_dir: Path = DEFAULT_COMBINED_RUNS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    target_budget: float = 0.2,
    batch_size: int = 128,
) -> dict:
    overall_start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("CROSS-DETECTOR ADVERSARIAL TRANSFERABILITY ANALYSIS (N = 1,000, Budget = 20%)")
    print("=" * 100)
    print(f"  Single-Attack Runs Dir : {single_runs_dir}")
    print(f"  Combined Runs Dir      : {combined_runs_dir}")
    print(f"  Output Directory       : {output_dir}")
    print(f"  Target Max Budget      : {target_budget:.3f} ({target_budget * 100:.1f}%)")
    print(f"  Inference Batch Size   : {batch_size}")

    # 1. Load trained detectors
    detectors = load_all_detectors(batch_size=batch_size)

    # 2. Extract budget == 0.2 rows from all 42 raw_results.csv files
    print(f"[2/4] Extracting budget == {target_budget} perturbed samples from 42 benchmark runs ...")
    t_extract = time.time()

    # run_data[(source_det, attack_key)] -> dict[sample_idx, row_dict]
    run_data: dict[tuple[str, str], dict[int, dict]] = {}
    valid_sets: dict[str, set[int]] = {det: set() for det in DETECTOR_ORDER}
    clean_confidences: dict[str, dict[int, float]] = {det: {} for det in DETECTOR_ORDER}
    clean_texts: dict[int, str] = {}

    total_runs = len(DETECTOR_ORDER) * len(ALL_ATTACK_KEYS)
    loaded_count = 0

    for source_det in DETECTOR_ORDER:
        for att_key in ALL_ATTACK_KEYS:
            loaded_count += 1
            if att_key == "combine_all":
                csv_path = combined_runs_dir / f"{source_det}_{att_key}" / "raw_results.csv"
            else:
                csv_path = single_runs_dir / f"{source_det}_{att_key}" / "raw_results.csv"

            records = extract_budget_rows(csv_path, target_budget=target_budget)
            run_data[(source_det, att_key)] = records

            # Record & verify deterministic sample_idx alignment per detector
            idx_set = set(records.keys())
            if not valid_sets[source_det]:
                valid_sets[source_det] = idx_set
            else:
                if valid_sets[source_det] != idx_set:
                    raise ValueError(
                        f"Sample index mismatch for {source_det} in {att_key}: "
                        f"expected {len(valid_sets[source_det])}, got {len(idx_set)}"
                    )

            for s_idx, rec in records.items():
                if s_idx not in clean_confidences[source_det]:
                    clean_confidences[source_det][s_idx] = rec["original_confidence"]
                if s_idx not in clean_texts:
                    clean_texts[s_idx] = rec["original_text"]

            if loaded_count % 14 == 0 or loaded_count == total_runs:
                elapsed = time.time() - t_extract
                rate = loaded_count / elapsed if elapsed > 0 else 0.0
                eta = (total_runs - loaded_count) / rate if rate > 0 else 0.0
                print(
                    f"  [{loaded_count:2d}/{total_runs:2d}] Loaded {source_det} runs "
                    f"({len(idx_set)} valid samples/run) | elapsed={elapsed:.2f}s | ETA={format_duration(eta)}"
                )

    # Seed per-detector inference cache with verified clean sample predictions
    # (and verify on a sample slice that live detector probabilities match original_confidence)
    pred_caches: dict[str, dict[str, tuple[int, float]]] = {det: {} for det in DETECTOR_ORDER}
    for det_key in DETECTOR_ORDER:
        det_valid_indices = sorted(valid_sets[det_key])
        check_indices = det_valid_indices[:16]
        check_texts = [clean_texts[i] for i in check_indices]
        _, check_probs, _, _ = batched_predict_malicious_proba_and_label(
            det_key, detectors[det_key], check_texts, batch_size=batch_size, cache=None
        )
        max_clean_diff = max(
            abs(check_probs[k] - clean_confidences[det_key][check_indices[k]])
            for k in range(len(check_indices))
        )
        print(
            f"  Verified {DETECTOR_SPECS[det_key]['short_name']:15s}: "
            f"|S_valid| = {len(det_valid_indices):3d} samples | clean prob verification max_diff = {max_clean_diff:.2e}"
        )
        # Seed clean texts for this detector's valid set into its cache
        for s_idx in det_valid_indices:
            pred_caches[det_key][clean_texts[s_idx]] = (1, clean_confidences[det_key][s_idx])

    # Print pairwise intersection sizes |S_valid(O_S) cap S_valid(f_T)|
    print("\n  Pairwise Valid Sample Intersections |S_valid(O_S) cap S_valid(f_T)|:")
    intersection_indices: dict[tuple[str, str], list[int]] = {}
    for s_det in DETECTOR_ORDER:
        row_strs = []
        for t_det in DETECTOR_ORDER:
            shared = sorted(valid_sets[s_det] & valid_sets[t_det])
            intersection_indices[(s_det, t_det)] = shared
            row_strs.append(
                f"{DETECTOR_SPECS[s_det]['abbr']}->{DETECTOR_SPECS[t_det]['abbr']}: {len(shared):3d}"
            )
        print("    " + "  |  ".join(row_strs))

    # 3. Evaluate all 14 attacks x 3 source oracles x 3 target detectors (126 cells, 84 off-diagonal transfer evaluations)
    print(f"\n[3/4] Running Batched Cross-Detector Transfer Inference (14 attacks x 3 sources x 3 targets) ...")

    # Group off-diagonal evaluations by target detector or iterate in canonical order
    # Pre-evaluating unique perturbed texts per target detector in large batches or run-by-run with ETA
    total_off_diag_pairs = len(ALL_ATTACK_KEYS) * len(DETECTOR_ORDER) * (len(DETECTOR_ORDER) - 1)  # 84
    total_cells = len(ALL_ATTACK_KEYS) * len(DETECTOR_ORDER) * len(DETECTOR_ORDER)  # 126

    # Estimate work weight per target detector for accurate ETA (Transformer on GPU is ~35x slower than CharCNN and ~350x slower than LR)
    target_weights = {
        "logistic_regression": 0.08,
        "char_cnn": 0.50,
        "transformer": 12.0,
    }
    total_weight = sum(
        target_weights[t_det]
        for _att in ALL_ATTACK_KEYS
        for s_det in DETECTOR_ORDER
        for t_det in DETECTOR_ORDER
        if s_det != t_det
    )
    completed_weight = 0.0
    actual_time_by_target = {det: 0.0 for det in DETECTOR_ORDER}
    runs_done_by_target = {det: 0 for det in DETECTOR_ORDER}

    t_eval_start = time.time()
    off_diag_idx = 0

    # Store results: cell_results[(att_key, source_det, target_det)] = dict
    cell_results: dict[tuple[str, str, str], dict] = {}

    for att_idx, att_key in enumerate(ALL_ATTACK_KEYS, start=1):
        att_info = ATTACK_METADATA[att_key]
        for source_det in DETECTOR_ORDER:
            source_records = run_data[(source_det, att_key)]
            source_valid_list = sorted(source_records.keys())
            num_source_valid = len(source_valid_list)
            source_evaded_set = {
                idx for idx, rec in source_records.items() if rec["evaded"] == 1
            }
            num_source_evaded = len(source_evaded_set)
            source_direct_evasion_rate = (
                num_source_evaded / num_source_valid if num_source_valid > 0 else 0.0
            )

            for target_det in DETECTOR_ORDER:
                shared_indices = intersection_indices[(source_det, target_det)]
                num_shared = len(shared_indices)
                is_direct = source_det == target_det

                if is_direct:
                    # Direct white-box evaluation (diagonal f_T == O_S)
                    target_orig_confs = [
                        source_records[idx]["original_confidence"] for idx in shared_indices
                    ]
                    target_pert_confs = [
                        source_records[idx]["perturbed_confidence"] for idx in shared_indices
                    ]
                    target_evaded_flags = [
                        source_records[idx]["evaded"] for idx in shared_indices
                    ]
                    cache_hits = num_shared
                    num_inferred = 0
                    step_elapsed = 0.0
                else:
                    # Cross-detector transfer evaluation (off-diagonal f_T != O_S)
                    off_diag_idx += 1
                    t_step = time.time()
                    pert_texts_shared = [
                        source_records[idx]["perturbed_text"] for idx in shared_indices
                    ]

                    preds, probs_mal, cache_hits, num_inferred = (
                        batched_predict_malicious_proba_and_label(
                            target_det,
                            detectors[target_det],
                            pert_texts_shared,
                            batch_size=batch_size,
                            cache=pred_caches[target_det],
                        )
                    )

                    target_orig_confs = [
                        clean_confidences[target_det][idx] for idx in shared_indices
                    ]
                    target_pert_confs = probs_mal
                    target_evaded_flags = [1 if p != 1 else 0 for p in preds]

                    step_elapsed = time.time() - t_step
                    actual_time_by_target[target_det] += step_elapsed
                    runs_done_by_target[target_det] += 1
                    completed_weight += target_weights[target_det]

                # a) Overall Transfer Evasion Rate & Mean Confidence Drop across S_valid(O_S) cap S_valid(f_T)
                num_overall_evaded = sum(target_evaded_flags)
                overall_evasion_rate = (
                    num_overall_evaded / num_shared if num_shared > 0 else 0.0
                )
                overall_detection_rate = 1.0 - overall_evasion_rate
                se_overall_evasion = (
                    math.sqrt(overall_evasion_rate * (1.0 - overall_evasion_rate) / num_shared)
                    if num_shared > 0
                    else 0.0
                )

                mean_orig_conf = (
                    sum(target_orig_confs) / num_shared if num_shared > 0 else 0.0
                )
                mean_pert_conf = (
                    sum(target_pert_confs) / num_shared if num_shared > 0 else 0.0
                )
                std_pert_conf = (
                    statistics.stdev(target_pert_confs) if num_shared > 1 else 0.0
                )
                mean_conf_drop = mean_orig_conf - mean_pert_conf

                # b) Conditional Transfer Success Rate among S_evaded(O_S) cap S_valid(f_T)
                cond_positions = [
                    pos for pos, idx in enumerate(shared_indices) if idx in source_evaded_set
                ]
                num_cond_denom = len(cond_positions)
                num_cond_evaded = sum(target_evaded_flags[pos] for pos in cond_positions)
                cond_transfer_rate = (
                    num_cond_evaded / num_cond_denom if num_cond_denom > 0 else 0.0
                )
                cond_mean_conf_drop = (
                    sum(
                        target_orig_confs[pos] - target_pert_confs[pos]
                        for pos in cond_positions
                    )
                    / num_cond_denom
                    if num_cond_denom > 0
                    else 0.0
                )

                source_evasion_in_shared_rate = (
                    num_cond_denom / num_shared if num_shared > 0 else 0.0
                )

                cell_dict = {
                    "attack_key": att_key,
                    "attack_name": att_info["display_name"],
                    "category": att_info["category"],
                    "source_detector": source_det,
                    "source_detector_name": DETECTOR_SPECS[source_det]["display_name"],
                    "target_detector": target_det,
                    "target_detector_name": DETECTOR_SPECS[target_det]["display_name"],
                    "pair_type": "direct" if is_direct else "transfer",
                    "budget": target_budget,
                    "source_valid_samples": num_source_valid,
                    "target_valid_samples": len(valid_sets[target_det]),
                    "shared_valid_samples": num_shared,
                    "source_direct_evaded": num_source_evaded,
                    "source_direct_evasion_pct": round(source_direct_evasion_rate * 100, 2),
                    "source_evaded_in_shared": num_cond_denom,
                    "source_direct_evasion_in_shared_pct": round(
                        source_evasion_in_shared_rate * 100, 2
                    ),
                    "overall_transfer_evaded": num_overall_evaded,
                    "overall_transfer_evasion_rate": round(overall_evasion_rate, 6),
                    "overall_transfer_evasion_pct": round(overall_evasion_rate * 100, 2),
                    "overall_transfer_detection_pct": round(overall_detection_rate * 100, 2),
                    "se_overall_transfer_evasion_pct": round(se_overall_evasion * 100, 2),
                    "mean_target_orig_confidence": round(mean_orig_conf, 6),
                    "mean_target_pert_confidence": round(mean_pert_conf, 6),
                    "std_target_pert_confidence": round(std_pert_conf, 6),
                    "mean_confidence_drop": round(mean_conf_drop, 6),
                    "mean_conf_drop_pct": round(mean_conf_drop * 100, 2),
                    "conditional_transfer_evaded": num_cond_evaded,
                    "conditional_transfer_denom": num_cond_denom,
                    "conditional_transfer_success_rate": round(cond_transfer_rate, 6),
                    "conditional_transfer_success_pct": round(cond_transfer_rate * 100, 2),
                    "conditional_mean_conf_drop_pct": round(cond_mean_conf_drop * 100, 2),
                }
                cell_results[(att_key, source_det, target_det)] = cell_dict

                if not is_direct:
                    # Compute adaptive ETA using measured per-target speed where available
                    total_elapsed = time.time() - t_eval_start
                    rem_sec = 0.0
                    for td in DETECTOR_ORDER:
                        total_td_runs = len(ALL_ATTACK_KEYS) * (len(DETECTOR_ORDER) - 1)  # 28
                        rem_td_runs = total_td_runs - runs_done_by_target[td]
                        if runs_done_by_target[td] > 0:
                            avg_td_sec = actual_time_by_target[td] / runs_done_by_target[td]
                        else:
                            avg_td_sec = target_weights[td]
                        rem_sec += rem_td_runs * avg_td_sec

                    pair_label = (
                        f"{DETECTOR_SPECS[source_det]['abbr']}->{DETECTOR_SPECS[target_det]['abbr']}"
                    )
                    print(
                        f"  [{off_diag_idx:2d}/{total_off_diag_pairs:2d}] "
                        f"{att_key:16s} | {pair_label:13s} | "
                        f"Ev={overall_evasion_rate * 100:5.1f}% ({num_overall_evaded:3d}/{num_shared:3d}) | "
                        f"ΔConf={-mean_conf_drop * 100:+6.2f}% | "
                        f"Cond={cond_transfer_rate * 100:5.1f}% ({num_cond_evaded:3d}/{num_cond_denom:3d}) | "
                        f"inferred={num_inferred:3d} (cached={cache_hits:3d}) in {step_elapsed:4.2f}s | "
                        f"ETA={format_duration(rem_sec)}"
                    )

    total_eval_time = time.time() - t_eval_start
    print(
        f"\n  Completed all {total_cells} matrix cells ({total_off_diag_pairs} off-diagonal transfer evaluations) "
        f"in {format_duration(total_eval_time)} ({total_eval_time:.2f}s)."
    )

    # 4. Write Outputs: transfer_matrix.csv, transfer_summary.txt, transfer_results.json
    print(f"\n[4/4] Generating Summary Files in {output_dir} ...")

    matrix_csv_path = output_dir / "transfer_matrix.csv"
    summary_txt_path = output_dir / "transfer_summary.txt"
    results_json_path = output_dir / "transfer_results.json"

    # 4a. Write transfer_matrix.csv
    csv_fieldnames = [
        "attack_key",
        "attack_name",
        "category",
        "source_detector",
        "source_detector_name",
        "target_detector",
        "target_detector_name",
        "pair_type",
        "budget",
        "source_valid_samples",
        "target_valid_samples",
        "shared_valid_samples",
        "source_direct_evaded",
        "source_direct_evasion_pct",
        "source_evaded_in_shared",
        "source_direct_evasion_in_shared_pct",
        "overall_transfer_evaded",
        "overall_transfer_evasion_rate",
        "overall_transfer_evasion_pct",
        "overall_transfer_detection_pct",
        "se_overall_transfer_evasion_pct",
        "mean_target_orig_confidence",
        "mean_target_pert_confidence",
        "std_target_pert_confidence",
        "mean_confidence_drop",
        "mean_conf_drop_pct",
        "conditional_transfer_evaded",
        "conditional_transfer_denom",
        "conditional_transfer_success_rate",
        "conditional_transfer_success_pct",
        "conditional_mean_conf_drop_pct",
    ]

    # Rank attacks by mean off-diagonal overall transfer evasion rate
    attack_rankings = []
    for att_key in ALL_ATTACK_KEYS:
        off_diag_cells = [
            cell_results[(att_key, s, t)]
            for s in DETECTOR_ORDER
            for t in DETECTOR_ORDER
            if s != t
        ]
        diag_cells = [
            cell_results[(att_key, d, d)]
            for d in DETECTOR_ORDER
        ]
        mean_off_ev = sum(c["overall_transfer_evasion_pct"] for c in off_diag_cells) / len(off_diag_cells)
        mean_off_drop = sum(c["mean_conf_drop_pct"] for c in off_diag_cells) / len(off_diag_cells)
        # For conditional transfer rate, compute both unweighted mean across pairs with denom > 0
        # and pooled conditional rate (total conditional evaded / total conditional denom)
        active_cond_cells = [c for c in off_diag_cells if c["conditional_transfer_denom"] > 0]
        mean_off_cond = (
            sum(c["conditional_transfer_success_pct"] for c in active_cond_cells) / len(active_cond_cells)
            if active_cond_cells
            else 0.0
        )
        pooled_cond_evaded = sum(c["conditional_transfer_evaded"] for c in off_diag_cells)
        pooled_cond_denom = sum(c["conditional_transfer_denom"] for c in off_diag_cells)
        pooled_off_cond = (
            (pooled_cond_evaded / pooled_cond_denom * 100.0) if pooled_cond_denom > 0 else 0.0
        )

        mean_diag_ev = sum(c["overall_transfer_evasion_pct"] for c in diag_cells) / len(diag_cells)
        mean_diag_drop = sum(c["mean_conf_drop_pct"] for c in diag_cells) / len(diag_cells)
        max_off_ev = max(c["overall_transfer_evasion_pct"] for c in off_diag_cells)

        attack_rankings.append({
            "attack_key": att_key,
            "attack_name": ATTACK_METADATA[att_key]["display_name"],
            "category": ATTACK_METADATA[att_key]["category"],
            "mean_off_diag_evasion_pct": round(mean_off_ev, 2),
            "mean_off_diag_conf_drop_pct": round(mean_off_drop, 2),
            "mean_off_diag_conditional_pct": round(mean_off_cond, 2),
            "pooled_off_diag_conditional_pct": round(pooled_off_cond, 2),
            "pooled_off_diag_conditional_evaded": pooled_cond_evaded,
            "pooled_off_diag_conditional_denom": pooled_cond_denom,
            "max_off_diag_evasion_pct": round(max_off_ev, 2),
            "mean_direct_evasion_pct": round(mean_diag_ev, 2),
            "mean_direct_conf_drop_pct": round(mean_diag_drop, 2),
            "transfer_retention_ratio_pct": round(
                (mean_off_ev / mean_diag_ev * 100.0) if mean_diag_ev > 0 else 0.0, 2
            ),
        })

    attack_rankings.sort(
        key=lambda x: (
            x["mean_off_diag_evasion_pct"],
            x["mean_off_diag_conf_drop_pct"],
            x["mean_direct_evasion_pct"],
        ),
        reverse=True,
    )
    ranked_attack_keys = [a["attack_key"] for a in attack_rankings]

    with open(matrix_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        for att_key in ranked_attack_keys:
            for s_det in DETECTOR_ORDER:
                for t_det in DETECTOR_ORDER:
                    writer.writerow(cell_results[(att_key, s_det, t_det)])

    # Helper to build aggregate 3x3 matrix across a subset of attack keys
    def build_aggregate_3x3(att_keys: list[str]) -> dict[tuple[str, str], dict]:
        agg = {}
        for s_det in DETECTOR_ORDER:
            for t_det in DETECTOR_ORDER:
                cells = [cell_results[(k, s_det, t_det)] for k in att_keys]
                n_att = len(cells)
                mean_ev = sum(c["overall_transfer_evasion_pct"] for c in cells) / n_att if n_att else 0.0
                mean_drp = sum(c["mean_conf_drop_pct"] for c in cells) / n_att if n_att else 0.0
                active_c = [c for c in cells if c["conditional_transfer_denom"] > 0]
                mean_cond = (
                    sum(c["conditional_transfer_success_pct"] for c in active_c) / len(active_c)
                    if active_c
                    else 0.0
                )
                tot_cond_ev = sum(c["conditional_transfer_evaded"] for c in cells)
                tot_cond_den = sum(c["conditional_transfer_denom"] for c in cells)
                pooled_cond = (tot_cond_ev / tot_cond_den * 100.0) if tot_cond_den > 0 else 0.0
                tot_ov_ev = sum(c["overall_transfer_evaded"] for c in cells)
                tot_ov_den = sum(c["shared_valid_samples"] for c in cells)
                agg[(s_det, t_det)] = {
                    "source_detector": s_det,
                    "target_detector": t_det,
                    "pair_type": "direct" if s_det == t_det else "transfer",
                    "shared_valid_samples_per_attack": len(intersection_indices[(s_det, t_det)]),
                    "num_attacks_averaged": n_att,
                    "mean_overall_evasion_pct": round(mean_ev, 2),
                    "mean_conf_drop_pct": round(mean_drp, 2),
                    "mean_conditional_success_pct": round(mean_cond, 2),
                    "pooled_conditional_success_pct": round(pooled_cond, 2),
                    "total_overall_evaded": tot_ov_ev,
                    "total_overall_evaluated": tot_ov_den,
                    "total_conditional_evaded": tot_cond_ev,
                    "total_conditional_denom": tot_cond_den,
                }
        return agg

    agg_13_single = build_aggregate_3x3(SINGLE_ATTACK_KEYS)
    agg_combine_all = build_aggregate_3x3(["combine_all"])
    agg_all_14 = build_aggregate_3x3(ALL_ATTACK_KEYS)

    category_keys = {
        cat: [k for k in ALL_ATTACK_KEYS if ATTACK_METADATA[k]["category"] == cat]
        for cat in ["structure", "metadata", "content", "combined"]
    }
    agg_by_category = {cat: build_aggregate_3x3(keys) for cat, keys in category_keys.items()}

    # 4b. Build transfer_summary.txt
    summary_text = build_transfer_summary_text(
        valid_sets=valid_sets,
        intersection_indices=intersection_indices,
        cell_results=cell_results,
        attack_rankings=attack_rankings,
        agg_13_single=agg_13_single,
        agg_combine_all=agg_combine_all,
        agg_all_14=agg_all_14,
        agg_by_category=agg_by_category,
        target_budget=target_budget,
        total_runtime=time.time() - overall_start,
    )
    summary_txt_path.write_text(summary_text, encoding="utf-8")

    # 4c. Build transfer_results.json
    results_payload = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "single_attacks_dir": str(single_runs_dir),
            "combined_runs_dir": str(combined_runs_dir),
            "output_dir": str(output_dir),
            "budget": target_budget,
            "num_total_samples": 1000,
            "seed": 3073,
            "num_attacks": len(ALL_ATTACK_KEYS),
            "num_detectors": len(DETECTOR_ORDER),
            "num_runs_loaded": total_runs,
            "num_matrix_cells": total_cells,
            "num_off_diagonal_pairs": total_off_diag_pairs,
            "runtime_seconds": round(time.time() - overall_start, 2),
            "valid_sample_counts": {
                det: len(valid_sets[det]) for det in DETECTOR_ORDER
            },
            "pairwise_shared_valid_counts": {
                f"{s}->{t}": len(intersection_indices[(s, t)])
                for s in DETECTOR_ORDER
                for t in DETECTOR_ORDER
            },
        },
        "aggregate_matrices": {
            "single_13_attacks": {
                f"{s}->{t}": agg_13_single[(s, t)]
                for s in DETECTOR_ORDER
                for t in DETECTOR_ORDER
            },
            "combine_all": {
                f"{s}->{t}": agg_combine_all[(s, t)]
                for s in DETECTOR_ORDER
                for t in DETECTOR_ORDER
            },
            "all_14_attacks": {
                f"{s}->{t}": agg_all_14[(s, t)]
                for s in DETECTOR_ORDER
                for t in DETECTOR_ORDER
            },
            "by_category": {
                cat: {
                    f"{s}->{t}": agg_by_category[cat][(s, t)]
                    for s in DETECTOR_ORDER
                    for t in DETECTOR_ORDER
                }
                for cat in ["structure", "metadata", "content", "combined"]
            },
        },
        "attack_rankings": attack_rankings,
        "per_attack_matrices": {
            att_key: {
                f"{s}->{t}": cell_results[(att_key, s, t)]
                for s in DETECTOR_ORDER
                for t in DETECTOR_ORDER
            }
            for att_key in ranked_attack_keys
        },
    }

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"  + Saved CSV matrix   : {matrix_csv_path}")
    print(f"  + Saved TXT summary  : {summary_txt_path}")
    print(f"  + Saved JSON results : {results_json_path}")
    print("\n" + summary_text)
    return results_payload


def _format_3x3_table(
    title: str,
    agg_matrix: dict[tuple[str, str], dict],
    intersection_indices: dict[tuple[str, str], list[int]],
) -> list[str]:
    """Formats a 3x3 Source Oracle (rows) x Target Detector (cols) ASCII table."""
    lines = []
    lines.append("=" * 124)
    lines.append(title)
    lines.append(
        "Rows = Source Oracle (O_S) | Columns = Target Detector (f_T) | "
        "Diagonal [Direct] = White-Box Oracle | Off-Diagonal = Transfer"
    )
    lines.append(
        "Cell Format: Overall Evasion% (-ΔConf%) | Cond: Conditional% (Pooled%) [n=shared_valid]"
    )
    lines.append("=" * 124)

    col_headers = [
        f"{'Source Oracle (O_S) \\ Target (f_T)':34s}",
        f"{'LogReg (TF-IDF)':32s}",
        f"{'CharCNN':32s}",
        f"{'Transformer (DistilBERT)':32s}",
        f"{'Off-Diag Source Mean':24s}",
    ]
    hdr = " | ".join(col_headers)
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for s_det in DETECTOR_ORDER:
        s_label = f"{DETECTOR_SPECS[s_det]['display_name']} (n={len(intersection_indices[(s_det, s_det)])})"
        row_cols = [f"{s_label:34s}"]
        off_evs = []
        off_drps = []
        off_pconds = []
        for t_det in DETECTOR_ORDER:
            cell = agg_matrix[(s_det, t_det)]
            ev = cell["mean_overall_evasion_pct"]
            drp = cell["mean_conf_drop_pct"]
            pcond = cell["pooled_conditional_success_pct"]
            if s_det == t_det:
                c_str = f"[Direct] {ev:5.1f}% ({-drp:+6.2f}%)"
            else:
                c_str = f"{ev:5.1f}% ({-drp:+6.2f}%) | Cond:{pcond:5.1f}%"
                off_evs.append(ev)
                off_drps.append(drp)
                off_pconds.append(pcond)
            row_cols.append(f"{c_str:32s}")

        s_mean_ev = sum(off_evs) / len(off_evs) if off_evs else 0.0
        s_mean_drp = sum(off_drps) / len(off_drps) if off_drps else 0.0
        s_mean_cond = sum(off_pconds) / len(off_pconds) if off_pconds else 0.0
        s_mean_str = f"{s_mean_ev:5.1f}% ({-s_mean_drp:+6.2f}%) C:{s_mean_cond:4.1f}%"
        row_cols.append(f"{s_mean_str:24s}")
        lines.append(" | ".join(row_cols))

    lines.append("-" * len(hdr))
    # Target vulnerability footer (off-diagonal average incoming transfer to each target detector)
    t_footer = [f"{'Off-Diag Target Mean (Incoming)':34s}"]
    all_off_ev = []
    all_off_drp = []
    all_off_cond = []
    for t_det in DETECTOR_ORDER:
        in_cells = [agg_matrix[(s_det, t_det)] for s_det in DETECTOR_ORDER if s_det != t_det]
        t_ev = sum(c["mean_overall_evasion_pct"] for c in in_cells) / len(in_cells)
        t_drp = sum(c["mean_conf_drop_pct"] for c in in_cells) / len(in_cells)
        t_cond = sum(c["pooled_conditional_success_pct"] for c in in_cells) / len(in_cells)
        all_off_ev.extend(c["mean_overall_evasion_pct"] for c in in_cells)
        all_off_drp.extend(c["mean_conf_drop_pct"] for c in in_cells)
        all_off_cond.extend(c["pooled_conditional_success_pct"] for c in in_cells)
        f_str = f"{t_ev:5.1f}% ({-t_drp:+6.2f}%) | Cond:{t_cond:5.1f}%"
        t_footer.append(f"{f_str:32s}")

    g_ev = sum(all_off_ev) / len(all_off_ev) if all_off_ev else 0.0
    g_drp = sum(all_off_drp) / len(all_off_drp) if all_off_drp else 0.0
    g_cond = sum(all_off_cond) / len(all_off_cond) if all_off_cond else 0.0
    g_str = f"{g_ev:5.1f}% ({-g_drp:+6.2f}%) C:{g_cond:4.1f}%"
    t_footer.append(f"{g_str:24s}")
    lines.append(" | ".join(t_footer))
    lines.append("=" * 124)
    return lines


def build_transfer_summary_text(
    valid_sets: dict[str, set[int]],
    intersection_indices: dict[tuple[str, str], list[int]],
    cell_results: dict[tuple[str, str, str], dict],
    attack_rankings: list[dict],
    agg_13_single: dict[tuple[str, str], dict],
    agg_combine_all: dict[tuple[str, str], dict],
    agg_all_14: dict[tuple[str, str], dict],
    agg_by_category: dict[str, dict[tuple[str, str], dict]],
    target_budget: float,
    total_runtime: float,
) -> str:
    lines: list[str] = []
    lines.append("=" * 142)
    lines.append(
        f"CROSS-DETECTOR ADVERSARIAL TRANSFERABILITY BENCHMARK — EXECUTIVE SUMMARY (N = 1,000, Budget = {target_budget * 100:.0f}%)"
    )
    lines.append("=" * 142)

    # Top single & combined attacks
    single_rankings = [a for a in attack_rankings if a["attack_key"] != "combine_all"]
    comb_ranking = next((a for a in attack_rankings if a["attack_key"] == "combine_all"), None)
    best_single_ev = single_rankings[0] if single_rankings else None
    best_single_cond = (
        max(single_rankings, key=lambda x: x["pooled_off_diag_conditional_pct"])
        if single_rankings
        else None
    )

    if comb_ranking:
        lines.append(
            f"  * Combined Best-of-K Transferability : {comb_ranking['attack_name']} [{comb_ranking['category'].upper()}] "
            f"— {comb_ranking['mean_off_diag_evasion_pct']:.1f}% Mean Off-Diag Transfer Evasion "
            f"(-{comb_ranking['mean_off_diag_conf_drop_pct']:.2f}% ΔConf, "
            f"{comb_ranking['pooled_off_diag_conditional_pct']:.1f}% Conditional Transfer, "
            f"Peak Pair: {comb_ranking['max_off_diag_evasion_pct']:.1f}%)"
        )
    if best_single_ev:
        lines.append(
            f"  * Most Transferable Single Attack    : {best_single_ev['attack_name']} [{best_single_ev['category'].upper()}] "
            f"— {best_single_ev['mean_off_diag_evasion_pct']:.1f}% Mean Off-Diag Transfer Evasion "
            f"(-{best_single_ev['mean_off_diag_conf_drop_pct']:.2f}% ΔConf, "
            f"{best_single_ev['pooled_off_diag_conditional_pct']:.1f}% Conditional Transfer, "
            f"Peak Pair: {best_single_ev['max_off_diag_evasion_pct']:.1f}%)"
        )
    if best_single_cond:
        lines.append(
            f"  * Highest Conditional Transfer Rate  : {best_single_cond['attack_name']} [{best_single_cond['category'].upper()}] "
            f"— {best_single_cond['pooled_off_diag_conditional_pct']:.1f}% Pooled Conditional Transfer "
            f"({best_single_cond['pooled_off_diag_conditional_evaded']}/{best_single_cond['pooled_off_diag_conditional_denom']} source-evading samples also evade target)"
        )

    lines.append(
        f"  * Valid Sample Alignment (ε = 0)     : "
        f"LogReg={len(valid_sets['logistic_regression'])}/1000 | "
        f"CharCNN={len(valid_sets['char_cnn'])}/1000 | "
        f"Transformer={len(valid_sets['transformer'])}/1000 | "
        f"Shared Intersections: LR∩CNN={len(intersection_indices[('logistic_regression', 'char_cnn')])}, "
        f"LR∩Trans={len(intersection_indices[('logistic_regression', 'transformer')])}, "
        f"CNN∩Trans={len(intersection_indices[('char_cnn', 'transformer')])}"
    )
    lines.append(f"  * Total Evaluation Runtime           : {total_runtime:.2f}s")

    # Aggregate 3x3 tables
    lines.append("")
    lines.extend(
        _format_3x3_table(
            "3x3 CROSS-DETECTOR TRANSFER MATRIX — COMBINED BEST-OF-K ATTACK (combine_all, Budget = 20%)",
            agg_combine_all,
            intersection_indices,
        )
    )

    lines.append("")
    lines.extend(
        _format_3x3_table(
            "3x3 CROSS-DETECTOR TRANSFER MATRIX — AVERAGED ACROSS 13 SINGLE ATTACKS (Budget = 20%)",
            agg_13_single,
            intersection_indices,
        )
    )

    lines.append("")
    lines.extend(
        _format_3x3_table(
            "3x3 CROSS-DETECTOR TRANSFER MATRIX — AVERAGED ACROSS ALL 14 ATTACKS (13 Single + Combined, Budget = 20%)",
            agg_all_14,
            intersection_indices,
        )
    )

    # Category summary
    lines.append("\n" + "=" * 142)
    lines.append("TRANSFERABILITY BY ATTACK CATEGORY (Averaged Across 6 Off-Diagonal Source -> Target Pairs vs. 3 Direct Pairs)")
    lines.append("=" * 142)
    cat_hdr = (
        f"{'Category':14s} | {'# Attacks':9s} | {'Mean Direct Ev (ΔConf)':24s} | "
        f"{'Mean Off-Diag Transfer Ev (ΔConf)':33s} | {'Pooled Cond. Transfer':23s} | {'Transfer/Direct Ratio':21s}"
    )
    lines.append(cat_hdr)
    lines.append("-" * len(cat_hdr))
    for cat in ["combined", "structure", "content", "metadata"]:
        c_agg = agg_by_category[cat]
        diag_c = [c_agg[(d, d)] for d in DETECTOR_ORDER]
        off_c = [c_agg[(s, t)] for s in DETECTOR_ORDER for t in DETECTOR_ORDER if s != t]
        n_att = diag_c[0]["num_attacks_averaged"] if diag_c else 0
        d_ev = sum(c["mean_overall_evasion_pct"] for c in diag_c) / len(diag_c)
        d_drp = sum(c["mean_conf_drop_pct"] for c in diag_c) / len(diag_c)
        o_ev = sum(c["mean_overall_evasion_pct"] for c in off_c) / len(off_c)
        o_drp = sum(c["mean_conf_drop_pct"] for c in off_c) / len(off_c)
        tot_cev = sum(c["total_conditional_evaded"] for c in off_c)
        tot_cden = sum(c["total_conditional_denom"] for c in off_c)
        p_cond = (tot_cev / tot_cden * 100.0) if tot_cden > 0 else 0.0
        ratio = (o_ev / d_ev * 100.0) if d_ev > 0 else 0.0
        lines.append(
            f"{cat.upper():14s} | {n_att:9d} | {f'{d_ev:5.1f}% ({-d_drp:+6.2f}%)':24s} | "
            f"{f'{o_ev:5.1f}% ({-o_drp:+6.2f}%)':33s} | "
            f"{f'{p_cond:5.1f}% ({tot_cev}/{tot_cden})':23s} | {f'{ratio:5.1f}%':21s}"
        )
    lines.append("=" * 142)

    # Full 14-Attack x 6 Directed Transfer Pairs Table — Metric (a): Overall Transfer Evasion Rate & ΔConf
    off_pairs = [
        ("logistic_regression", "char_cnn", "LR -> CharCNN"),
        ("logistic_regression", "transformer", "LR -> Trans"),
        ("char_cnn", "logistic_regression", "CharCNN -> LR"),
        ("char_cnn", "transformer", "CharCNN -> Trans"),
        ("transformer", "logistic_regression", "Trans -> LR"),
        ("transformer", "char_cnn", "Trans -> CharCNN"),
    ]

    lines.append("\n" + "=" * 184)
    lines.append(
        "METRIC (A): OVERALL TRANSFER EVASION RATE & MEAN CONFIDENCE DROP (ΔConf) ACROSS S_valid(O_S) ∩ S_valid(f_T)"
    )
    lines.append("Ranked by Mean Off-Diagonal Transfer Evasion Rate | Cell Format: Evasion% (-ΔConf%)")
    lines.append("=" * 184)

    cols_a = [
        f"{'#':>2s}",
        f"{'Attack Algorithm':30s}",
        f"{'Category':10s}",
    ]
    for _s, _t, plabel in off_pairs:
        cols_a.append(f"{plabel:18s}")
    cols_a.append(f"{'Mean Transfer (ΔConf)':21s}")
    cols_a.append(f"{'Mean Direct (ΔConf)':20s}")
    hdr_a = " | ".join(cols_a)
    lines.append(hdr_a)
    lines.append("-" * len(hdr_a))

    for rank_idx, astat in enumerate(attack_rankings, start=1):
        att_key = astat["attack_key"]
        row_a = [
            f"{rank_idx:2d}",
            f"{astat['attack_name'][:30]:30s}",
            f"{astat['category'].upper():10s}",
        ]
        for s_det, t_det, _plabel in off_pairs:
            c = cell_results[(att_key, s_det, t_det)]
            ev = c["overall_transfer_evasion_pct"]
            drp = c["mean_conf_drop_pct"]
            row_a.append(f"{f'{ev:5.1f}% ({-drp:+6.2f}%)':18s}")
        m_off = f"{astat['mean_off_diag_evasion_pct']:5.1f}% ({-astat['mean_off_diag_conf_drop_pct']:+6.2f}%)"
        m_dir = f"{astat['mean_direct_evasion_pct']:5.1f}% ({-astat['mean_direct_conf_drop_pct']:+6.2f}%)"
        row_a.append(f"{m_off:21s}")
        row_a.append(f"{m_dir:20s}")
        lines.append(" | ".join(row_a))

    lines.append("-" * len(hdr_a))
    # Footer averages for Single (13) and All (14)
    for label, subset_keys in [
        ("13 SINGLE ATTACKS AVERAGE", SINGLE_ATTACK_KEYS),
        ("ALL 14 ATTACKS AVERAGE", ALL_ATTACK_KEYS),
    ]:
        f_row = [f"{'':2s}", f"{label:30s}", f"{'AVG':10s}"]
        pair_evs = []
        pair_drps = []
        for s_det, t_det, _plabel in off_pairs:
            p_ev = sum(cell_results[(k, s_det, t_det)]["overall_transfer_evasion_pct"] for k in subset_keys) / len(subset_keys)
            p_drp = sum(cell_results[(k, s_det, t_det)]["mean_conf_drop_pct"] for k in subset_keys) / len(subset_keys)
            pair_evs.append(p_ev)
            pair_drps.append(p_drp)
            f_row.append(f"{f'{p_ev:5.1f}% ({-p_drp:+6.2f}%)':18s}")
        m_ev = sum(pair_evs) / len(pair_evs)
        m_drp = sum(pair_drps) / len(pair_drps)
        d_ev = sum(
            cell_results[(k, d, d)]["overall_transfer_evasion_pct"]
            for k in subset_keys
            for d in DETECTOR_ORDER
        ) / (len(subset_keys) * len(DETECTOR_ORDER))
        d_drp = sum(
            cell_results[(k, d, d)]["mean_conf_drop_pct"]
            for k in subset_keys
            for d in DETECTOR_ORDER
        ) / (len(subset_keys) * len(DETECTOR_ORDER))
        f_row.append(f"{f'{m_ev:5.1f}% ({-m_drp:+6.2f}%)':21s}")
        f_row.append(f"{f'{d_ev:5.1f}% ({-d_drp:+6.2f}%)':20s}")
        lines.append(" | ".join(f_row))
    lines.append("=" * 184)

    # Full 14-Attack x 6 Directed Transfer Pairs Table — Metric (b): Conditional Transfer Success Rate
    lines.append("\n" + "=" * 184)
    lines.append(
        "METRIC (B): CONDITIONAL TRANSFER SUCCESS RATE AMONG S_evaded(O_S) ∩ S_valid(f_T) (Source-Evading Samples Only)"
    )
    lines.append("Same Ranking Order | Cell Format: Cond% (evaded_both / evaded_source_in_shared)")
    lines.append("=" * 184)

    cols_b = [
        f"{'#':>2s}",
        f"{'Attack Algorithm':30s}",
        f"{'Category':10s}",
    ]
    for _s, _t, plabel in off_pairs:
        cols_b.append(f"{plabel:18s}")
    cols_b.append(f"{'Pooled Cond. Rate':21s}")
    cols_b.append(f"{'Mean Pair Cond. Rate':20s}")
    hdr_b = " | ".join(cols_b)
    lines.append(hdr_b)
    lines.append("-" * len(hdr_b))

    for rank_idx, astat in enumerate(attack_rankings, start=1):
        att_key = astat["attack_key"]
        row_b = [
            f"{rank_idx:2d}",
            f"{astat['attack_name'][:30]:30s}",
            f"{astat['category'].upper():10s}",
        ]
        for s_det, t_det, _plabel in off_pairs:
            c = cell_results[(att_key, s_det, t_det)]
            cpct = c["conditional_transfer_success_pct"]
            cev = c["conditional_transfer_evaded"]
            cden = c["conditional_transfer_denom"]
            if cden > 0:
                c_str = f"{cpct:5.1f}% ({cev:3d}/{cden:3d})"
            else:
                c_str = f"  N/A (  0/  0)"
            row_b.append(f"{c_str:18s}")
        p_str = (
            f"{astat['pooled_off_diag_conditional_pct']:5.1f}% "
            f"({astat['pooled_off_diag_conditional_evaded']}/{astat['pooled_off_diag_conditional_denom']})"
        )
        m_str = f"{astat['mean_off_diag_conditional_pct']:5.1f}%"
        row_b.append(f"{p_str:21s}")
        row_b.append(f"{m_str:20s}")
        lines.append(" | ".join(row_b))

    lines.append("-" * len(hdr_b))
    for label, subset_keys in [
        ("13 SINGLE ATTACKS POOLED", SINGLE_ATTACK_KEYS),
        ("ALL 14 ATTACKS POOLED", ALL_ATTACK_KEYS),
    ]:
        f_row = [f"{'':2s}", f"{label:30s}", f"{'POOLED':10s}"]
        tot_ev_all = 0
        tot_den_all = 0
        pair_conds = []
        for s_det, t_det, _plabel in off_pairs:
            pev = sum(cell_results[(k, s_det, t_det)]["conditional_transfer_evaded"] for k in subset_keys)
            pden = sum(cell_results[(k, s_det, t_det)]["conditional_transfer_denom"] for k in subset_keys)
            tot_ev_all += pev
            tot_den_all += pden
            ppct = (pev / pden * 100.0) if pden > 0 else 0.0
            pair_conds.append(ppct)
            f_row.append(f"{f'{ppct:5.1f}% ({pev}/{pden})':18s}")
        g_pooled = (tot_ev_all / tot_den_all * 100.0) if tot_den_all > 0 else 0.0
        g_mean = sum(pair_conds) / len(pair_conds) if pair_conds else 0.0
        f_row.append(f"{f'{g_pooled:5.1f}% ({tot_ev_all}/{tot_den_all})':21s}")
        f_row.append(f"{f'{g_mean:5.1f}%':20s}")
        lines.append(" | ".join(f_row))
    lines.append("=" * 184)

    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-Detector Adversarial Transferability Analysis at Max Budget.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--single-runs-dir",
        type=str,
        default=str(DEFAULT_SINGLE_RUNS_DIR),
        help="Path to the 39 single-attack benchmark runs directory.",
    )
    parser.add_argument(
        "--combined-runs-dir",
        type=str,
        default=str(DEFAULT_COMBINED_RUNS_DIR),
        help="Path to the 3 combine_all benchmark runs directory.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for transfer_matrix.csv, transfer_summary.txt, and transfer_results.json.",
    )
    parser.add_argument(
        "--budget",
        "-b",
        type=float,
        default=0.20,
        help="Perturbation budget step to extract from raw_results.csv.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for neural detector GPU/CPU inference.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_transferability_analysis(
        single_runs_dir=Path(args.single_runs_dir),
        combined_runs_dir=Path(args.combined_runs_dir),
        output_dir=Path(args.output_dir),
        target_budget=args.budget,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
