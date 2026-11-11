#!/usr/bin/env python3
"""
Adversarial Degradation Report Generator
=========================================
Generates a rich, interactive, standalone HTML report for adversarial
robustness evaluations and degradation curve analysis.

Supports:
- Multi-detector runs (e.g. CharCNN, Logistic Regression, Transformer)
- Single-detector runs
- Interactive degradation curves & multi-model overlay
- GitHub-style git-diff visualizer (Unified, Split Side-by-Side, Substitutions, Patch)
- Comprehensive sample-level filtering, search, sorting, and raw data inspector
- Heatmap timeline of sample flip thresholds
"""

import os
import sys
import json
import csv
import base64
import re
import math
import html
import difflib
import argparse
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# Text Tokenization & Diff Utilities
# ---------------------------------------------------------------------------
def get_word_tokens_with_spaces(text: str) -> list[str]:
    """Tokenize text into words along with their trailing whitespace to preserve formatting while diffing clean words."""
    tokens = re.findall(r"\S+\s*", text)
    if not tokens and text:
        return [text]
    return tokens


def compute_word_diff(orig_text: str, pert_text: str):
    """
    Computes a fine-grained token-level diff between original and perturbed texts.
    Performs SequenceMatcher on clean word tokens (so whitespace differences do not
    throw off alignment), then associates trailing whitespace with each token.
    Splits 1-to-1 word replacements cleanly (e.g. 'job' -> 'tasks').
    """
    tokens1 = get_word_tokens_with_spaces(orig_text)
    tokens2 = get_word_tokens_with_spaces(pert_text)

    words1 = [t.strip() for t in tokens1]
    words2 = [t.strip() for t in tokens2]

    sm = difflib.SequenceMatcher(None, words1, words2)
    chunks = []
    substitutions = []
    words_changed = 0

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            chunks.append(["=", "".join(tokens1[i1:i2])])
        elif tag == "replace":
            len1 = i2 - i1
            len2 = j2 - j1
            words_changed += max(len1, len2)

            # If equal number of words replaced, pair each word 1-to-1 cleanly
            if len1 == len2:
                for k in range(len1):
                    chunks.append(["~", tokens1[i1 + k], tokens2[j1 + k]])
                    if words1[i1 + k] != words2[j1 + k]:
                        substitutions.append({
                            "orig": words1[i1 + k],
                            "pert": words2[j1 + k]
                        })
            else:
                chunks.append(["~", "".join(tokens1[i1:i2]), "".join(tokens2[j1:j2])])
                substitutions.append({
                    "orig": " ".join(words1[i1:i2]),
                    "pert": " ".join(words2[j1:j2])
                })
        elif tag == "delete":
            words_changed += (i2 - i1)
            chunks.append(["-", "".join(tokens1[i1:i2])])
            substitutions.append({
                "orig": " ".join(words1[i1:i2]),
                "pert": ""
            })
        elif tag == "insert":
            words_changed += (j2 - j1)
            chunks.append(["+", "".join(tokens2[j1:j2])])
            substitutions.append({
                "orig": "",
                "pert": " ".join(words2[j1:j2])
            })

    # Generate unified git diff patch
    orig_lines = orig_text.splitlines(keepends=True)
    pert_lines = pert_text.splitlines(keepends=True)
    patch_lines = list(difflib.unified_diff(
        orig_lines,
        pert_lines,
        fromfile="original.txt",
        tofile="perturbed.txt",
        lineterm=""
    ))
    patch = "\n".join(patch_lines)

    return chunks, substitutions, words_changed, patch


# ---------------------------------------------------------------------------
# Directory Resolution & Data Loading
# ---------------------------------------------------------------------------
def resolve_results_directory(path_str: str) -> tuple[Path, list[Path]]:
    """
    Determines if path is:
    1. The analysis/results root folder (picks newest timestamp run)
    2. A timestamp folder with detector subfolders
    3. A single detector folder with raw_results.csv
    Returns: (base_dir, list of detector directories)
    """
    p = Path(path_str).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Specified path does not exist: {p}")

    # Case 1: If directory has timestamp subfolders (e.g. results/), pick the newest timestamp run
    timestamp_subdirs = [
        d for d in p.iterdir()
        if d.is_dir() and re.match(r"^\d{8}-\d{6}$", d.name)
    ]
    if timestamp_subdirs:
        latest = sorted(timestamp_subdirs, key=lambda d: d.name)[-1]
        print(f"Detected results root. Automatically selecting latest timestamp run: {latest.name}")
        return resolve_results_directory(str(latest))

    # Case 2: Multi-detector timestamp directory (contains subdirs with raw_results.csv)
    subdirs_with_results = [
        d for d in p.iterdir()
        if d.is_dir() and (d / "raw_results.csv").exists()
    ]
    if subdirs_with_results:
        return p, sorted(subdirs_with_results, key=lambda d: d.name)

    # Case 3: Direct single detector folder
    if (p / "raw_results.csv").exists():
        return p.parent, [p]

    raise ValueError(f"No benchmark results (raw_results.csv) found under {p}")


ATTACK_CATEGORY_MAP = {
    "textfooler": "content",
    "zero_width": "structure",
    "html_tag": "structure",
    "html_entity": "structure",
    "dom_wrapper": "structure",
    "homoglyph": "structure",
    "html_comment": "structure",
    "sender_header": "metadata",
    "auth_results": "metadata",
    "received_routing": "metadata",
    "x_header": "metadata",
    "subject_metadata": "metadata",
    "timestamp": "metadata",
}

KNOWN_DETECTORS = {
    "logistic_regression": "Logistic Regression (TF-IDF)",
    "char_cnn": "CharCNN",
    "transformer": "Transformer (DistilBERT)",
}


def _infer_detector_and_attacker(config: dict, folder_name: str) -> dict:
    """
    Extracts normalized detector identity, attacker identity, and taxonomy
    category from `run_config.json` and folder name (supporting both legacy
    single-attacker and multi-detector × multi-attacker runs).
    """
    att_key = str(config.get("attacker_key", "")).strip() or "unknown"
    att_name = str(config.get("attacker_display_name", "")).strip() or att_key.replace("_", " ").title()

    base_det_key = str(config.get("base_detector_key", "")).strip()
    base_det_name = str(config.get("base_detector_display_name", "")).strip()

    raw_det_key = str(config.get("detector_key", folder_name)).strip()
    raw_det_name = str(config.get("detector_display_name", folder_name)).strip()

    if not base_det_key:
        if att_key != "unknown" and raw_det_key.endswith(f"_{att_key}"):
            base_det_key = raw_det_key[: -(len(att_key) + 1)]
        else:
            for k in KNOWN_DETECTORS:
                if raw_det_key == k or raw_det_key.startswith(f"{k}_"):
                    base_det_key = k
                    if att_key == "unknown" and raw_det_key.startswith(f"{k}_"):
                        att_key = raw_det_key[len(k) + 1 :]
                    break
        if not base_det_key:
            base_det_key = raw_det_key

    if not base_det_name:
        if " - " in raw_det_name:
            base_det_name = raw_det_name.split(" - ", 1)[0].strip()
        elif base_det_key in KNOWN_DETECTORS:
            base_det_name = KNOWN_DETECTORS[base_det_key]
        else:
            base_det_name = raw_det_name

    category = str(config.get("attacker_category", "")).strip().lower()
    if not category or category == "unknown":
        category = ATTACK_CATEGORY_MAP.get(att_key, "other")

    return {
        "detector_id": base_det_key,
        "detector_label": base_det_name,
        "attacker_id": att_key,
        "attacker_label": att_name,
        "attack_category": category,
    }


def load_detector_run(detector_dir: Path) -> dict:
    """Loads all data, metrics, curves, and raw results for a single detector run."""
    print(f"Loading detector results from {detector_dir.name} ...")

    # 1. Configuration
    config_file = detector_dir / "run_config.json"
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {
            "detector_key": detector_dir.name,
            "detector_display_name": detector_dir.name.replace("_", " ").title(),
            "timestamp": detector_dir.parent.name,
            "attacker_key": "unknown",
            "attacker_display_name": "Adversarial Attacker",
            "num_samples_requested": 0,
            "num_samples_valid": 0,
            "seed": 42,
            "max_budget": 0.2,
            "budget_steps": 20,
            "data_path": "texts.json",
            "model_path": ""
        }

    identity = _infer_detector_and_attacker(config, detector_dir.name)

    # 2. Degradation Metrics
    metrics_file = detector_dir / "degradation_metrics.csv"
    metrics = []
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                metrics.append({
                    "budget": float(row.get("budget", 0)),
                    "detection_rate": float(row.get("detection_rate", 0)),
                    "se_detection_rate": float(row.get("se_detection_rate", 0)),
                    "evasion_rate": float(row.get("evasion_rate", 0)),
                    "mean_orig_confidence": float(row.get("mean_orig_confidence", 0)),
                    "mean_pert_confidence": float(row.get("mean_pert_confidence", 0)),
                    "std_pert_confidence": float(row.get("std_pert_confidence", 0)),
                    "mean_confidence_drop": float(row.get("mean_confidence_drop", 0)),
                    "mean_real_budget_consumed": float(row.get("mean_real_budget_consumed", 0)),
                    "failed_transformations": int(float(row.get("failed_transformations", 0)))
                })

    # 3. Curve Image (if exists, embed as base64)
    curve_png = detector_dir / "degradation_curve.png"
    curve_b64 = ""
    if curve_png.exists():
        with open(curve_png, "rb") as f:
            curve_b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")

    # 4. Raw Results
    raw_file = detector_dir / "raw_results.csv"
    raw_rows = []
    sample_texts_cache = {}  # sample_idx -> orig_text
    sample_word_counts = {}

    # Track flip budgets per sample
    sample_flip_points = {}  # sample_idx -> min budget where evaded == 1

    with open(raw_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        for r in reader:
            s_idx = int(r.get("sample_idx", 0))
            b_val = float(r.get("budget", 0.0))
            evaded = int(r.get("evaded", 0))

            # Decode original text (cached once per sample)
            if s_idx not in sample_texts_cache:
                orig_b64 = r.get("original_text_b64", "")
                if orig_b64:
                    orig_decoded = base64.b64decode(orig_b64).decode("utf-8", errors="replace")
                else:
                    orig_decoded = f"[Sample #{s_idx} text not recorded in legacy CSV]"
                sample_texts_cache[s_idx] = orig_decoded
                sample_word_counts[s_idx] = len(orig_decoded.split())

            orig_text = sample_texts_cache[s_idx]
            pert_b64 = r.get("perturbed_text_b64", "")
            if pert_b64:
                pert_text = base64.b64decode(pert_b64).decode("utf-8", errors="replace")
            else:
                pert_text = orig_text

            # Track earliest flip
            if evaded == 1:
                if s_idx not in sample_flip_points or b_val < sample_flip_points[s_idx]:
                    sample_flip_points[s_idx] = b_val

            # Compute fine diff
            chunks, subs, w_changed, patch = compute_word_diff(orig_text, pert_text)

            orig_conf = float(r.get("original_confidence", 1.0))
            pert_conf = float(r.get("perturbed_confidence", 0.0))
            conf_drop = orig_conf - pert_conf

            raw_rows.append({
                "budget": b_val,
                "sample_idx": s_idx,
                "orig_pred": int(r.get("original_pred", 1)),
                "pert_pred": int(r.get("perturbed_pred", 1 - evaded)),
                "orig_conf": round(orig_conf, 4),
                "pert_conf": round(pert_conf, 4),
                "conf_drop": round(conf_drop, 4),
                "evaded": evaded,
                "real_budget_consumed": int(float(r.get("real_budget_consumed", w_changed))),
                "success": r.get("transformation_success", "true").strip().lower() == "true",
                "words_changed": w_changed,
                "chunks": chunks,
                "subs": subs,
                "patch": patch,
                "orig_text": orig_text,
                "pert_text": pert_text
            })

    # Summary Statistics
    total_valid = len(sample_texts_cache)
    final_metric = metrics[-1] if metrics else {}
    final_evasion_rate = final_metric.get("evasion_rate", 0.0)
    final_det_rate = final_metric.get("detection_rate", 0.0)

    # Average flip threshold
    flipped_budgets = list(sample_flip_points.values())
    avg_flip_budget = (sum(flipped_budgets) / len(flipped_budgets)) if flipped_budgets else None

    # Global word substitutions count
    sub_freq = {}
    for r in raw_rows:
        for sub in r["subs"]:
            pair = f"{sub['orig']} -> {sub['pert']}"
            sub_freq[pair] = sub_freq.get(pair, 0) + 1

    top_subs = sorted(sub_freq.items(), key=lambda x: x[1], reverse=True)[:15]

    return {
        "key": config.get("detector_key", detector_dir.name),
        "name": config.get("detector_display_name", detector_dir.name.replace("_", " ").title()),
        "detector_id": identity["detector_id"],
        "detector_label": identity["detector_label"],
        "attacker_id": identity["attacker_id"],
        "attacker_label": identity["attacker_label"],
        "attack_category": identity["attack_category"],
        "config": config,
        "metrics": metrics,
        "curve_b64": curve_b64,
        "raw_rows": raw_rows,
        "sample_texts": sample_texts_cache,
        "sample_word_counts": sample_word_counts,
        "sample_flip_points": sample_flip_points,
        "summary": {
            "total_samples": total_valid,
            "total_iterations": len(raw_rows),
            "final_detection_rate": round(final_det_rate * 100, 1),
            "final_evasion_rate": round(final_evasion_rate * 100, 1),
            "total_evaded_samples": len(sample_flip_points),
            "resilient_samples": total_valid - len(sample_flip_points),
            "avg_flip_budget": round(avg_flip_budget * 100, 1) if avg_flip_budget is not None else None,
            "mean_confidence_drop": round(final_metric.get("mean_confidence_drop", 0.0) * 100, 1),
            "avg_budget_consumed": round(final_metric.get("mean_real_budget_consumed", 0.0), 1),
            "total_failed_transforms": sum(1 for r in raw_rows if not r["success"]),
            "top_substitutions": top_subs
        }
    }


def generate_summary_tables(base_dir: Path, detectors_data: list[dict]) -> tuple[str, Path]:
    """
    Generates a comprehensive plain-text summary report (`summary_table.txt` and `table.txt`)
    and a 2D Attack × Detector pivot CSV (`summary_matrix.csv`) in `base_dir`.
    Returns `(summary_text, matrix_csv_path)`.
    """
    # Collect unique detectors and attackers in canonical order
    det_order = ["logistic_regression", "char_cnn", "transformer"]
    seen_dets = {}
    seen_atts = {}
    grid = {}  # (att_id, det_id) -> run dict

    for d in detectors_data:
        det_id = d["detector_id"]
        att_id = d["attacker_id"]
        seen_dets[det_id] = d["detector_label"]
        seen_atts[att_id] = (d["attacker_label"], d["attack_category"])
        grid[(att_id, det_id)] = d

    sorted_det_ids = sorted(
        seen_dets.keys(),
        key=lambda k: (det_order.index(k) if k in det_order else 99, k),
    )

    # Compute per-attacker aggregate metrics across detectors
    attacker_stats = []
    for att_id, (att_label, att_cat) in seen_atts.items():
        ev_vals = []
        det_vals = []
        drop_vals = []
        for det_id in sorted_det_ids:
            run = grid.get((att_id, det_id))
            if run:
                ev_vals.append(run["summary"]["final_evasion_rate"])
                det_vals.append(run["summary"]["final_detection_rate"])
                drop_vals.append(run["summary"]["mean_confidence_drop"])
        mean_ev = sum(ev_vals) / len(ev_vals) if ev_vals else 0.0
        mean_det = sum(det_vals) / len(det_vals) if det_vals else 100.0
        mean_drop = sum(drop_vals) / len(drop_vals) if drop_vals else 0.0
        max_ev = max(ev_vals) if ev_vals else 0.0
        attacker_stats.append({
            "attacker_id": att_id,
            "attacker_label": att_label,
            "category": att_cat,
            "mean_evasion": round(mean_ev, 1),
            "mean_detection": round(mean_det, 1),
            "mean_drop": round(mean_drop, 2),
            "max_evasion": round(max_ev, 1),
        })

    # Sort attackers by mean_evasion desc, then mean_drop desc
    attacker_stats.sort(key=lambda x: (x["mean_evasion"], x["mean_drop"]), reverse=True)

    # Write summary_matrix.csv
    matrix_csv_path = base_dir / "summary_matrix.csv"
    csv_headers = ["rank", "attacker_key", "attacker_name", "category"]
    for det_id in sorted_det_ids:
        csv_headers.extend([
            f"{det_id}_evasion_pct",
            f"{det_id}_detection_pct",
            f"{det_id}_conf_drop_pct",
        ])
    csv_headers.extend(["mean_evasion_pct", "mean_detection_pct", "mean_conf_drop_pct"])

    with open(matrix_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        for rank_idx, astat in enumerate(attacker_stats, start=1):
            att_id = astat["attacker_id"]
            row_dict = {
                "rank": rank_idx,
                "attacker_key": att_id,
                "attacker_name": astat["attacker_label"],
                "category": astat["category"],
                "mean_evasion_pct": astat["mean_evasion"],
                "mean_detection_pct": astat["mean_detection"],
                "mean_conf_drop_pct": astat["mean_drop"],
            }
            for det_id in sorted_det_ids:
                run = grid.get((att_id, det_id))
                if run:
                    row_dict[f"{det_id}_evasion_pct"] = run["summary"]["final_evasion_rate"]
                    row_dict[f"{det_id}_detection_pct"] = run["summary"]["final_detection_rate"]
                    row_dict[f"{det_id}_conf_drop_pct"] = run["summary"]["mean_confidence_drop"]
                else:
                    row_dict[f"{det_id}_evasion_pct"] = ""
                    row_dict[f"{det_id}_detection_pct"] = ""
                    row_dict[f"{det_id}_conf_drop_pct"] = ""
            writer.writerow(row_dict)

    # Build formatted plain-text summary report
    lines = []
    lines.append("=" * 110)
    lines.append("ADVERSARIAL ROBUSTNESS BENCHMARK — EXECUTIVE SUMMARY & 2D MATRIX")
    lines.append("=" * 110)

    if attacker_stats:
        best_ev = attacker_stats[0]
        best_drop = max(attacker_stats, key=lambda x: x["mean_drop"])
        lines.append(
            f"  * Most Effective Attack (Evasion) : {best_ev['attacker_label']} [{best_ev['category'].upper()}] "
            f"— {best_ev['mean_evasion']:.1f}% Mean Evasion (Peak: {best_ev['max_evasion']:.1f}%)"
        )
        lines.append(
            f"  * Largest Confidence Drop (ΔConf) : {best_drop['attacker_label']} [{best_drop['category'].upper()}] "
            f"— -{best_drop['mean_drop']:.2f}% Mean Confidence Drop"
        )

    # Detector averages
    det_summaries = []
    for det_id in sorted_det_ids:
        det_runs = [grid[(a["attacker_id"], det_id)] for a in attacker_stats if (a["attacker_id"], det_id) in grid]
        if det_runs:
            avg_ev = sum(r["summary"]["final_evasion_rate"] for r in det_runs) / len(det_runs)
            avg_det = sum(r["summary"]["final_detection_rate"] for r in det_runs) / len(det_runs)
            avg_drp = sum(r["summary"]["mean_confidence_drop"] for r in det_runs) / len(det_runs)
            worst_run = max(det_runs, key=lambda r: (r["summary"]["final_evasion_rate"], r["summary"]["mean_confidence_drop"]))
            det_summaries.append((det_id, seen_dets[det_id], avg_det, avg_ev, avg_drp, worst_run))

    if det_summaries:
        lines.append("\nDETECTOR VULNERABILITY SUMMARY (Averaged Across All Evaluated Attacks):")
        lines.append("-" * 110)
        for _did, dlabel, avg_det, avg_ev, avg_drp, wrun in det_summaries:
            lines.append(
                f"  * {dlabel:28s} | Avg Retained: {avg_det:5.1f}% | Avg Evasion: {avg_ev:5.1f}% | "
                f"Avg ΔConf: -{avg_drp:5.2f}% | Hardest Hit By: {wrun['attacker_label']} "
                f"({wrun['summary']['final_evasion_rate']:.1f}% ev, -{wrun['summary']['mean_confidence_drop']:.1f}% Δconf)"
            )

    # 2D Matrix Table: Attack x Detector at Max Budget
    lines.append("\n" + "=" * 110)
    lines.append("2D ATTACK × DETECTOR MATRIX (At Max Budget — Ranked by Mean Evasion Rate)")
    lines.append("Format per cell: Evasion% (-ΔConf%)")
    lines.append("=" * 110)

    short_det_names = {
        "logistic_regression": "LogReg (TF-IDF)",
        "char_cnn": "CharCNN",
        "transformer": "Transformer",
    }
    header_cols = [f"{'#':>2s}", f"{'Attack Algorithm':30s}", f"{'Category':10s}"]
    for det_id in sorted_det_ids:
        sname = short_det_names.get(det_id, seen_dets[det_id][:18])
        header_cols.append(f"{sname:20s}")
    header_cols.append(f"{'Mean Evasion (ΔConf)':22s}")
    header_line = " | ".join(header_cols)
    lines.append(header_line)
    lines.append("-" * len(header_line))

    for rank_idx, astat in enumerate(attacker_stats, start=1):
        att_id = astat["attacker_id"]
        row_cols = [
            f"{rank_idx:2d}",
            f"{astat['attacker_label'][:30]:30s}",
            f"{astat['category'].upper():10s}",
        ]
        for det_id in sorted_det_ids:
            run = grid.get((att_id, det_id))
            if run:
                ev = run["summary"]["final_evasion_rate"]
                drp = run["summary"]["mean_confidence_drop"]
                cell_str = f"{ev:5.1f}% ({-drp:+6.2f}%)"
            else:
                cell_str = "N/A"
            row_cols.append(f"{cell_str:20s}")
        mean_str = f"{astat['mean_evasion']:5.1f}% ({-astat['mean_drop']:+6.2f}%)"
        row_cols.append(f"{mean_str:22s}")
        lines.append(" | ".join(row_cols))

    lines.append("-" * len(header_line))
    # Average footer row
    avg_cols = [f"{'':2s}", f"{'DETECTOR AVERAGE':30s}", f"{'ALL':10s}"]
    for _did, _dlabel, _avg_det, avg_ev, avg_drp, _wrun in det_summaries:
        c_str = f"{avg_ev:5.1f}% ({-avg_drp:+6.2f}%)"
        avg_cols.append(f"{c_str:20s}")
    overall_ev = sum(x["mean_evasion"] for x in attacker_stats) / len(attacker_stats) if attacker_stats else 0.0
    overall_drp = sum(x["mean_drop"] for x in attacker_stats) / len(attacker_stats) if attacker_stats else 0.0
    avg_cols.append(f"{f'{overall_ev:5.1f}% ({-overall_drp:+6.2f}%)':22s}")
    lines.append(" | ".join(avg_cols))
    lines.append("=" * 110)

    summary_text = "\n".join(lines) + "\n"
    (base_dir / "summary_table.txt").write_text(summary_text, encoding="utf-8")
    (base_dir / "table.txt").write_text(summary_text, encoding="utf-8")
    return summary_text, matrix_csv_path


# ---------------------------------------------------------------------------
# HTML Template String (Non-f-string to preserve JS and CSS syntax)
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>__RUN_TITLE__ - Adversarial Robustness Report</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-canvas: #090d16;
      --bg-surface: #101625;
      --bg-surface-elevated: #162035;
      --bg-card: #131b2e;
      --border-subtle: #1e2c47;
      --border-accent: #2e436d;
      
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --text-dim: #64748b;

      --accent-primary: #6366f1;
      --accent-primary-hover: #4f46e5;
      --accent-cyan: #06b6d4;
      --accent-indigo: #818cf8;

      --evaded-red: #f43f5e;
      --evaded-bg: rgba(244, 63, 94, 0.12);
      --evaded-border: rgba(244, 63, 94, 0.35);

      --caught-green: #10b981;
      --caught-bg: rgba(16, 185, 129, 0.12);
      --caught-border: rgba(16, 185, 129, 0.35);

      --diff-ins-bg: rgba(16, 185, 129, 0.22);
      --diff-ins-text: #6ee7b7;
      --diff-ins-border: #10b981;

      --diff-del-bg: rgba(244, 63, 94, 0.22);
      --diff-del-text: #fda4af;
      --diff-del-border: #f43f5e;

      --radius-sm: 6px;
      --radius-md: 10px;
      --radius-lg: 16px;
      --shadow-card: 0 4px 20px -2px rgba(0, 0, 0, 0.45);
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      background-color: var(--bg-canvas);
      color: var(--text-main);
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
      padding-bottom: 80px;
    }

    /* Layout Container */
    .container {
      max-width: 1440px;
      margin: 0 auto;
      padding: 0 24px;
    }

    /* Top Sticky Header */
    header.top-header {
      position: sticky;
      top: 0;
      z-index: 50;
      background: rgba(9, 13, 22, 0.88);
      backdrop-filter: blur(14px);
      border-bottom: 1px solid var(--border-subtle);
      padding: 14px 0;
      margin-bottom: 28px;
    }

    .header-inner {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      flex-wrap: wrap;
    }

    .brand-title {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .brand-badge {
      background: linear-gradient(135deg, #4f46e5, #06b6d4);
      color: #fff;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      padding: 4px 8px;
      border-radius: var(--radius-sm);
      letter-spacing: 0.05em;
    }

    h1.app-title {
      font-size: 19px;
      font-weight: 700;
      color: #fff;
      letter-spacing: -0.01em;
    }

    .header-meta {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }

    .meta-tag {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      padding: 4px 10px;
      border-radius: var(--radius-sm);
      font-size: 12px;
      color: var(--text-muted);
    }
    .meta-tag strong {
      color: var(--text-main);
    }

    .header-actions {
      display: flex;
      gap: 8px;
    }

    .btn {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: var(--bg-surface-elevated);
      color: var(--text-main);
      border: 1px solid var(--border-subtle);
      padding: 7px 14px;
      border-radius: var(--radius-sm);
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.15s ease;
      text-decoration: none;
    }
    .btn:hover {
      background: var(--border-subtle);
      border-color: var(--border-accent);
      color: #fff;
    }
    .btn-primary {
      background: var(--accent-primary);
      border-color: var(--accent-primary);
      color: #fff;
    }
    .btn-primary:hover {
      background: var(--accent-primary-hover);
    }

    /* Global KPI Cards Grid */
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 16px;
      margin-bottom: 28px;
    }

    .kpi-card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 18px;
      box-shadow: var(--shadow-card);
      position: relative;
      overflow: hidden;
      transition: transform 0.2s, border-color 0.2s;
    }
    .kpi-card:hover {
      border-color: var(--border-accent);
      transform: translateY(-2px);
    }
    .kpi-card::before {
      content: "";
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 3px;
      background: var(--kpi-accent, var(--accent-primary));
    }

    .kpi-label {
      font-size: 12px;
      text-transform: uppercase;
      font-weight: 600;
      color: var(--text-muted);
      letter-spacing: 0.05em;
      margin-bottom: 6px;
    }

    .kpi-value {
      font-size: 28px;
      font-weight: 800;
      color: #fff;
      letter-spacing: -0.02em;
      display: flex;
      align-items: baseline;
      gap: 8px;
    }

    .kpi-subtext {
      font-size: 12px;
      color: var(--text-dim);
      margin-top: 4px;
    }

    /* Executive Insights Banner */
    .insights-banner {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
      margin-bottom: 24px;
    }
    .insight-card {
      background: linear-gradient(145deg, var(--bg-card), var(--bg-surface));
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 16px 18px;
      box-shadow: var(--shadow-card);
      cursor: pointer;
      transition: transform 0.15s, border-color 0.15s;
      position: relative;
    }
    .insight-card:hover {
      border-color: var(--border-accent);
      transform: translateY(-2px);
    }
    .insight-tag {
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--accent-cyan);
      margin-bottom: 6px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .insight-headline {
      font-size: 16px;
      font-weight: 700;
      color: #fff;
      margin-bottom: 4px;
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .insight-detail {
      font-size: 12px;
      color: var(--text-muted);
      line-height: 1.45;
    }

    /* Category Badges */
    .cat-badge {
      display: inline-flex;
      align-items: center;
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      padding: 2px 7px;
      border-radius: 4px;
    }
    .cat-structure {
      background: rgba(244, 63, 94, 0.16);
      color: #fb7185;
      border: 1px solid rgba(244, 63, 94, 0.35);
    }
    .cat-content {
      background: rgba(99, 102, 241, 0.18);
      color: #a5b4fc;
      border: 1px solid rgba(99, 102, 241, 0.38);
    }
    .cat-metadata {
      background: rgba(245, 158, 11, 0.16);
      color: #fcd34d;
      border: 1px solid rgba(245, 158, 11, 0.35);
    }

    /* 2D Attack x Detector Heatmap Matrix */
    .matrix-section {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      padding: 22px;
      margin-bottom: 24px;
      box-shadow: var(--shadow-card);
    }
    .matrix-controls {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }
    .matrix-table-wrapper {
      overflow-x: auto;
      margin-top: 14px;
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      background: var(--bg-canvas);
    }
    .matrix-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    .matrix-table th {
      background: var(--bg-surface-elevated);
      color: var(--text-muted);
      font-weight: 700;
      padding: 11px 14px;
      border-bottom: 1px solid var(--border-accent);
      border-right: 1px solid rgba(255, 255, 255, 0.04);
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: 0.04em;
      text-align: center;
      cursor: pointer;
      transition: background 0.15s, color 0.15s;
    }
    .matrix-table th:first-child,
    .matrix-table th:nth-child(2) {
      text-align: left;
    }
    .matrix-table th:hover {
      background: #1e2c47;
      color: #fff;
    }
    .matrix-table td {
      padding: 10px 14px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      border-right: 1px solid rgba(255, 255, 255, 0.04);
      vertical-align: middle;
    }
    .matrix-attack-cell {
      cursor: pointer;
      font-weight: 600;
      color: var(--text-main);
      white-space: nowrap;
      transition: color 0.15s;
    }
    .matrix-attack-cell:hover {
      color: var(--accent-cyan);
    }
    .matrix-rank-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 24px;
      height: 22px;
      border-radius: 4px;
      font-family: 'Fira Code', monospace;
      font-size: 11px;
      font-weight: 700;
      background: var(--bg-surface);
      color: var(--text-muted);
      border: 1px solid var(--border-subtle);
    }
    .matrix-rank-badge.top-rank {
      background: rgba(244, 63, 94, 0.2);
      color: #fda4af;
      border-color: rgba(244, 63, 94, 0.45);
    }
    .matrix-heat-cell {
      text-align: center;
      cursor: pointer;
      transition: transform 0.1s, outline 0.1s, filter 0.15s;
      position: relative;
      min-width: 145px;
    }
    .matrix-heat-cell:hover {
      outline: 2px solid #fff;
      z-index: 5;
      filter: brightness(1.15);
    }
    .matrix-heat-cell.selected-cell {
      outline: 2px solid var(--accent-cyan);
      box-shadow: inset 0 0 0 1px var(--accent-cyan);
    }
    .heat-main-val {
      font-size: 15px;
      font-weight: 800;
      color: #fff;
      font-family: 'Fira Code', monospace;
      line-height: 1.2;
    }
    .heat-sub-val {
      font-size: 11px;
      color: rgba(241, 245, 249, 0.78);
      margin-top: 2px;
      font-family: 'Fira Code', monospace;
    }
    .matrix-footer-row td {
      background: var(--bg-surface-elevated);
      font-weight: 700;
      border-top: 2px solid var(--border-accent);
    }

    /* Two-Level Dimensional Selector Bar */
    .dimension-nav-panel {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      padding: 14px 18px;
      margin-bottom: 24px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .dim-row {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .dim-label {
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-muted);
      min-width: 90px;
    }
    .dim-pills {
      display: flex;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
      flex: 1;
    }
    .detector-tab {
      padding: 6px 13px;
      border-radius: var(--radius-sm);
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid transparent;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 7px;
      white-space: nowrap;
      transition: all 0.15s ease;
    }
    .detector-tab:hover {
      color: #fff;
      background: rgba(255, 255, 255, 0.06);
      border-color: var(--border-subtle);
    }
    .detector-tab.active {
      background: var(--bg-card);
      color: #fff;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
      border: 1px solid var(--accent-primary);
    }
    .detector-tab-badge {
      font-size: 10px;
      padding: 1px 6px;
      border-radius: 10px;
      background: rgba(255, 255, 255, 0.08);
      font-family: 'Fira Code', monospace;
    }
    .detector-tab.active .detector-tab-badge {
      background: var(--accent-primary);
      color: #fff;
    }
    .active-slice-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-top: 8px;
      border-top: 1px solid var(--border-subtle);
      font-size: 12px;
      color: var(--text-muted);
      flex-wrap: wrap;
      gap: 8px;
    }

    /* Interactive Chart Legend */
    .chart-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
      padding-top: 12px;
      border-top: 1px solid var(--border-subtle);
    }
    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      padding: 4px 10px;
      border-radius: var(--radius-sm);
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      font-size: 12px;
      color: var(--text-main);
      cursor: pointer;
      transition: all 0.15s;
    }
    .legend-item:hover {
      border-color: var(--border-accent);
      background: var(--bg-surface-elevated);
    }
    .legend-swatch {
      width: 12px;
      height: 12px;
      border-radius: 3px;
      display: inline-block;
    }

    /* Multi-Detector Comparison Matrix */
    .comparison-section {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      padding: 22px;
      margin-bottom: 28px;
    }
    .section-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 18px;
    }
    .section-title {
      font-size: 16px;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .comparison-table-wrapper {
      overflow-x: auto;
    }
    .data-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      text-align: left;
    }
    .data-table th {
      background: var(--bg-surface);
      color: var(--text-muted);
      font-weight: 600;
      padding: 10px 14px;
      border-bottom: 1px solid var(--border-subtle);
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: 0.05em;
    }
    .data-table td {
      padding: 12px 14px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      color: var(--text-main);
    }
    .data-table tr:hover td {
      background: rgba(255, 255, 255, 0.02);
    }

    /* Interactive Chart Card */
    .chart-card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      padding: 22px;
      margin-bottom: 28px;
      box-shadow: var(--shadow-card);
    }

    .chart-toolbar {
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }

    .chart-view-toggle {
      display: inline-flex;
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      padding: 3px;
    }
    .chart-toggle-btn {
      padding: 4px 10px;
      font-size: 12px;
      font-weight: 500;
      color: var(--text-muted);
      background: transparent;
      border: none;
      border-radius: 4px;
      cursor: pointer;
    }
    .chart-toggle-btn.active {
      background: var(--bg-card);
      color: #fff;
      font-weight: 600;
    }

    .canvas-container {
      position: relative;
      width: 100%;
      height: 380px;
      margin-top: 14px;
    }
    canvas#chartCanvas {
      width: 100%;
      height: 100%;
      display: block;
    }
    .chart-tooltip {
      position: absolute;
      display: none;
      background: rgba(16, 22, 37, 0.95);
      border: 1px solid var(--border-accent);
      padding: 10px 14px;
      border-radius: var(--radius-sm);
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6);
      pointer-events: none;
      font-size: 12px;
      z-index: 20;
      backdrop-filter: blur(8px);
      white-space: nowrap;
    }

    /* Sample Trajectory Timeline Heatmap */
    .timeline-card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      padding: 22px;
      margin-bottom: 28px;
    }
    .timeline-scroll-container {
      --cell-w: 32px;
      overflow: auto;
      max-height: 480px;
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      background: var(--bg-canvas);
      position: relative;
    }
    .timeline-scroll-container::-webkit-scrollbar {
      width: 8px;
      height: 8px;
    }
    .timeline-scroll-container::-webkit-scrollbar-track {
      background: #090d16;
    }
    .timeline-scroll-container::-webkit-scrollbar-thumb {
      background: #25334e;
      border-radius: 4px;
    }
    .timeline-scroll-container::-webkit-scrollbar-thumb:hover {
      background: #3b4e75;
    }
    .timeline-table {
      display: inline-block;
      min-width: 100%;
      border-collapse: collapse;
      user-select: none;
    }
    .timeline-header-row {
      display: flex;
      position: sticky;
      top: 0;
      z-index: 25;
      background: var(--bg-surface-elevated);
      border-bottom: 1px solid var(--border-accent);
    }
    .timeline-corner-cell {
      position: sticky;
      left: 0;
      top: 0;
      z-index: 35;
      width: 140px;
      min-width: 140px;
      background: var(--bg-surface-elevated);
      border-right: 1px solid var(--border-accent);
      padding: 6px 12px;
      font-size: 11px;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .timeline-header-cells {
      display: flex;
      gap: 3px;
      padding: 4px 6px;
    }
    .timeline-header-cell {
      width: var(--cell-w, 32px);
      min-width: var(--cell-w, 32px);
      text-align: center;
      font-size: 10px;
      font-weight: 700;
      color: var(--accent-cyan);
      font-family: 'Fira Code', monospace;
      padding: 4px 0;
      white-space: nowrap;
      cursor: default;
    }
    .timeline-header-cell.is-tick {
      color: var(--text-dim);
      font-size: 14px;
      line-height: 12px;
    }
    .timeline-row {
      display: flex;
      align-items: center;
      border-bottom: 1px solid rgba(255, 255, 255, 0.03);
    }
    .timeline-row:hover {
      background: rgba(255, 255, 255, 0.03);
    }
    .timeline-sample-lbl {
      position: sticky;
      left: 0;
      z-index: 15;
      width: 140px;
      min-width: 140px;
      background: var(--bg-card);
      border-right: 1px solid var(--border-accent);
      padding: 6px 12px;
      font-size: 11px;
      font-weight: 600;
      color: var(--text-main);
      cursor: pointer;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .timeline-sample-lbl:hover {
      color: var(--accent-cyan);
      background: var(--bg-surface-elevated);
    }
    .timeline-cells {
      display: flex;
      gap: 3px;
      padding: 3px 6px;
    }
    .timeline-cell {
      width: var(--cell-w, 32px);
      min-width: var(--cell-w, 32px);
      height: 22px;
      border-radius: 2px;
      cursor: pointer;
      position: relative;
      transition: transform 0.1s, outline 0.1s;
      flex-shrink: 0;
    }
    .timeline-cell:hover {
      transform: scale(1.2);
      z-index: 12;
      outline: 2px solid #fff;
    }
    .cell-caught {
      background: rgba(16, 185, 129, 0.45);
      border: 1px solid #10b981;
    }
    .cell-evaded {
      background: rgba(244, 63, 94, 0.65);
      border: 1px solid #f43f5e;
    }
    .cell-failed {
      background: rgba(100, 116, 139, 0.25);
      border: 1px dashed #64748b;
    }
    .timeline-toolbar {
      display: flex;
      gap: 14px;
      align-items: center;
      flex-wrap: wrap;
    }
    .timeline-density-toggle {
      display: inline-flex;
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      padding: 2px;
      align-items: center;
    }
    .density-btn {
      padding: 3px 8px;
      font-size: 11px;
      font-weight: 600;
      color: var(--text-muted);
      background: transparent;
      border: none;
      border-radius: 3px;
      cursor: pointer;
    }
    .density-btn.active {
      background: var(--bg-card);
      color: #fff;
    }
    .timeline-footer {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 12px;
      font-size: 12px;
      color: var(--text-muted);
      flex-wrap: wrap;
      gap: 10px;
    }

    /* Filter & Search Toolbar */
    .filter-panel {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      padding: 18px 20px;
      margin-bottom: 24px;
      box-shadow: var(--shadow-card);
    }
    .filter-row {
      display: flex;
      gap: 14px;
      align-items: center;
      flex-wrap: wrap;
    }
    .search-box {
      flex: 1;
      min-width: 260px;
      position: relative;
    }
    .search-input {
      width: 100%;
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      padding: 8px 14px 8px 36px;
      color: #fff;
      font-size: 13px;
      font-family: inherit;
      outline: none;
      transition: border-color 0.15s;
    }
    .search-input:focus {
      border-color: var(--accent-primary);
    }
    .search-icon {
      position: absolute;
      left: 12px;
      top: 50%;
      transform: translateY(-50%);
      color: var(--text-dim);
      font-size: 14px;
    }

    .filter-select {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      padding: 8px 12px;
      color: var(--text-main);
      font-size: 13px;
      font-family: inherit;
      outline: none;
      cursor: pointer;
    }
    .filter-select:focus {
      border-color: var(--accent-primary);
    }

    .filter-stats-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 14px;
      padding-top: 12px;
      border-top: 1px solid var(--border-subtle);
      font-size: 12px;
      color: var(--text-muted);
      flex-wrap: wrap;
      gap: 10px;
    }

    /* Diff View Selector */
    .diff-mode-switcher {
      display: inline-flex;
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      padding: 2px;
    }
    .diff-mode-btn {
      padding: 4px 10px;
      font-size: 11px;
      font-weight: 600;
      color: var(--text-muted);
      background: transparent;
      border: none;
      border-radius: 4px;
      cursor: pointer;
    }
    .diff-mode-btn.active {
      background: var(--accent-primary);
      color: #fff;
    }

    /* Sample Cards List */
    .samples-list {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .sample-card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      overflow: hidden;
      box-shadow: var(--shadow-card);
      transition: border-color 0.15s;
    }
    .sample-card:hover {
      border-color: var(--border-accent);
    }
    .sample-card.is-evaded {
      border-left: 4px solid var(--evaded-red);
    }
    .sample-card.is-caught {
      border-left: 4px solid var(--caught-green);
    }

    .sample-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 14px 18px;
      background: var(--bg-surface);
      border-bottom: 1px solid var(--border-subtle);
      flex-wrap: wrap;
      gap: 12px;
    }

    .sample-title-area {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .sample-id-badge {
      font-family: 'Fira Code', monospace;
      font-size: 12px;
      font-weight: 600;
      background: var(--bg-canvas);
      padding: 3px 8px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border-subtle);
      color: #fff;
    }
    .budget-badge {
      font-size: 12px;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: var(--radius-sm);
      background: rgba(6, 182, 212, 0.15);
      color: var(--accent-cyan);
      border: 1px solid rgba(6, 182, 212, 0.3);
    }

    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: var(--radius-sm);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .status-evaded {
      background: var(--evaded-bg);
      color: var(--evaded-red);
      border: 1px solid var(--evaded-border);
    }
    .status-caught {
      background: var(--caught-bg);
      color: var(--caught-green);
      border: 1px solid var(--caught-border);
    }

    .sample-metrics-strip {
      display: flex;
      align-items: center;
      gap: 16px;
      flex-wrap: wrap;
    }

    .confidence-pill {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
    }

    .sample-body {
      padding: 16px 18px;
    }

    /* Git Diff Renderers */
    .diff-container {
      font-family: 'Fira Code', 'Courier New', monospace;
      font-size: 13px;
      line-height: 1.6;
      background: #0d121c;
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      padding: 14px;
      max-height: 380px;
      overflow-y: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }

    /* Inline Diff Highlights */
    .diff-del {
      background: var(--diff-del-bg);
      color: var(--diff-del-text);
      text-decoration: line-through;
      border-radius: 3px;
      padding: 1px 4px;
      margin: 0 1px;
      border: 1px solid var(--diff-del-border);
    }
    .diff-ins {
      background: var(--diff-ins-bg);
      color: var(--diff-ins-text);
      border-radius: 3px;
      padding: 1px 4px;
      margin: 0 1px;
      border: 1px solid var(--diff-ins-border);
      font-weight: 600;
    }

    /* Side by Side Diff */
    .split-diff-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    .split-pane {
      background: #0d121c;
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      padding: 12px;
      max-height: 380px;
      overflow-y: auto;
      font-family: 'Fira Code', monospace;
      font-size: 12px;
      line-height: 1.6;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .split-pane-header {
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 8px;
      padding-bottom: 6px;
      border-bottom: 1px solid var(--border-subtle);
      display: flex;
      justify-content: space-between;
    }

    /* Substitutions Table */
    .substitutions-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      font-family: 'Fira Code', monospace;
    }
    .substitutions-table th {
      background: var(--bg-surface);
      color: var(--text-muted);
      padding: 6px 10px;
      text-align: left;
      font-size: 11px;
      border-bottom: 1px solid var(--border-subtle);
    }
    .substitutions-table td {
      padding: 6px 10px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.03);
    }

    /* Unified Patch Monospace */
    .patch-block {
      background: #070a10;
      color: #94a3b8;
      font-family: 'Fira Code', monospace;
      font-size: 12px;
      line-height: 1.5;
      padding: 12px;
      border-radius: var(--radius-sm);
      overflow-x: auto;
      white-space: pre;
    }

    /* Card Footer Actions */
    .sample-footer {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 10px 18px;
      background: rgba(0, 0, 0, 0.2);
      border-top: 1px solid var(--border-subtle);
      font-size: 12px;
      color: var(--text-dim);
      flex-wrap: wrap;
      gap: 10px;
    }
    .sample-action-btns {
      display: flex;
      gap: 8px;
    }

    /* Pagination */
    .pagination-bar {
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 10px;
      margin-top: 30px;
    }

    /* Raw Data Modal */
    .modal-overlay {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.75);
      backdrop-filter: blur(8px);
      z-index: 100;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }
    .modal-overlay.active {
      display: flex;
    }
    .modal-dialog {
      background: var(--bg-card);
      border: 1px solid var(--border-accent);
      border-radius: var(--radius-lg);
      width: 100%;
      max-width: 860px;
      max-height: 85vh;
      display: flex;
      flex-direction: column;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.8);
      overflow: hidden;
    }
    .modal-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 16px 20px;
      border-bottom: 1px solid var(--border-subtle);
      background: var(--bg-surface);
    }
    .modal-title {
      font-size: 15px;
      font-weight: 700;
      color: #fff;
    }
    .modal-body {
      padding: 20px;
      overflow-y: auto;
      flex: 1;
    }
    .modal-code {
      background: #080d17;
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      padding: 14px;
      font-family: 'Fira Code', monospace;
      font-size: 12px;
      color: #cbd5e1;
      white-space: pre-wrap;
      word-break: break-all;
    }

    /* Toast Notification */
    .toast {
      position: fixed;
      bottom: 24px;
      right: 24px;
      background: #1e293b;
      border: 1px solid var(--accent-primary);
      color: #fff;
      padding: 10px 18px;
      border-radius: var(--radius-sm);
      box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
      font-size: 13px;
      font-weight: 500;
      opacity: 0;
      transform: translateY(10px);
      transition: all 0.2s ease;
      z-index: 120;
      pointer-events: none;
    }
    .toast.show {
      opacity: 1;
      transform: translateY(0);
    }
  </style>
</head>
<body>

  <!-- Top Sticky Navigation Bar -->
  <header class="top-header">
    <div class="container header-inner">
      <div class="brand-title">
        <span class="brand-badge">Adversarial Robustness</span>
        <h1 class="app-title">__RUN_TITLE__</h1>
      </div>

      <div class="header-meta">
        <span class="meta-tag">Detectors: <strong>__DETECTOR_SUMMARY__</strong></span>
        <span class="meta-tag">Attacks: <strong>__ATTACKER_NAME__</strong></span>
        <span class="meta-tag">Runs: <strong>__RUN_COUNT__</strong></span>
        <span class="meta-tag">Timestamp: <strong>__TIMESTAMP__</strong></span>
        <span class="meta-tag">Max Budget: <strong>__MAX_BUDGET_PCT__% (__BUDGET_STEPS__ steps)</strong></span>
        <span class="meta-tag">Seed: <strong>__SEED__</strong></span>
      </div>

      <div class="header-actions">
        <button class="btn btn-primary" onclick="exportMatrixCSV()">Export 2D Matrix CSV</button>
        <button class="btn" onclick="exportFilteredData('csv')">Export Iterations CSV</button>
        <button class="btn" onclick="exportFilteredData('json')">Export JSON</button>
      </div>
    </div>
  </header>

  <div class="container">

    <!-- Executive Insights Banner (Most effective attack, strongest confidence drop, detector comparison, category breakdown) -->
    <section class="insights-banner" id="insightsBanner" style="display: none;">
      <!-- Injected via JavaScript -->
    </section>

    <!-- 2D Attack x Detector Heatmap Matrix (Shown when multiple runs exist) -->
    <section class="matrix-section" id="matrixSection" style="display: none;">
      <div class="section-header" style="flex-wrap: wrap; gap: 12px;">
        <div>
          <h2 class="section-title">2D Attack &times; Detector Effectiveness Matrix</h2>
          <p class="kpi-subtext">At-a-glance comparison across all evaluated attack algorithms and detectors. Click any cell to drill into that exact (Detector, Attack) run, or click row/column headers to slice by Attack or Detector.</p>
        </div>
        <div class="matrix-controls">
          <select id="matrixMetricSelect" class="filter-select" onchange="renderMatrix()" title="Select metric displayed in heatmap cells">
            <option value="combined">Cell View: Evasion % + &Delta;Conf Drop %</option>
            <option value="evasion">Cell View: Evasion Rate (%)</option>
            <option value="detection">Cell View: Detection Retained (%)</option>
            <option value="drop">Cell View: Mean Confidence Drop (-&Delta;%)</option>
            <option value="flip">Cell View: Avg Flip Budget (%)</option>
          </select>

          <select id="matrixBudgetSelect" class="filter-select" onchange="renderMatrix()" title="Inspect matrix at a specific perturbation budget">
            <option value="max">Budget: Max (__MAX_BUDGET_PCT__%)</option>
          </select>

          <select id="matrixCategorySelect" class="filter-select" onchange="renderMatrix()" title="Filter rows by attack taxonomy category">
            <option value="all">Category: All Types</option>
            <option value="structure">Structure Attacks</option>
            <option value="content">Content Attacks</option>
            <option value="metadata">Metadata Attacks</option>
          </select>

          <select id="matrixSortSelect" class="filter-select" onchange="renderMatrix()" title="Sort attack rows">
            <option value="evasion_desc">Sort: Mean Evasion (High &rarr; Low)</option>
            <option value="drop_desc">Sort: Mean &Delta;Conf Drop (High &rarr; Low)</option>
            <option value="category">Sort: Attack Category</option>
            <option value="name_asc">Sort: Attack Name (A &rarr; Z)</option>
          </select>
        </div>
      </div>

      <div class="matrix-table-wrapper">
        <table class="matrix-table" id="matrixTable">
          <thead id="matrixThead"></thead>
          <tbody id="matrixTbody"></tbody>
          <tfoot id="matrixTfoot"></tfoot>
        </table>
      </div>
    </section>

    <!-- Two-Level Dimensional Navigation Bar (Detector Filter + Attack Filter) -->
    <nav class="dimension-nav-panel" id="detectorNav">
      <div class="dim-row" id="detectorRow">
        <span class="dim-label">1. Detector:</span>
        <div class="dim-pills" id="detectorPills"></div>
      </div>
      <div class="dim-row" id="attackerRow">
        <span class="dim-label">2. Attack:</span>
        <div class="dim-pills" id="attackerPills"></div>
      </div>
      <div class="active-slice-bar" id="activeSliceBar">
        <div id="activeSliceText">Viewing: <strong>All Detectors</strong> &times; <strong>All Attacks</strong></div>
        <button class="btn" id="btnResetSlice" style="padding: 3px 10px; font-size: 11px; display: none;" onclick="selectSlice('all', 'all')">Reset to Global Overview (All &times; All)</button>
      </div>
    </nav>

    <!-- KPI Summary Grid -->
    <section class="kpi-grid" id="kpiGrid">
      <!-- Injected via JavaScript based on selected detector/attack slice -->
    </section>

    <!-- Context-Aware Leaderboard & Robustness Comparison Table -->
    <section class="comparison-section" id="comparisonSection">
      <div class="section-header" style="flex-wrap: wrap; gap: 10px;">
        <div>
          <h2 class="section-title" id="comparisonTitle">Attack &amp; Detector Robustness Leaderboard</h2>
          <p class="kpi-subtext" id="comparisonSubtitle">Ranked by Evasion Rate (%) and Mean Confidence Drop (-&Delta;%) across __BUDGET_STEPS__ perturbation budget levels</p>
        </div>
        <div style="display: flex; gap: 8px; align-items: center;">
          <select id="comparisonGroupSelect" class="filter-select" style="padding: 5px 10px; font-size: 12px;" onchange="renderComparisonTable()">
            <option value="auto">Group: Auto (By Active Selection)</option>
            <option value="by_attack">Group by Attack (Mean Across Detectors)</option>
            <option value="by_detector">Group by Detector (Mean Across Attacks)</option>
            <option value="all_runs">Show All Individual Runs (__RUN_COUNT__)</option>
          </select>
        </div>
      </div>
      <div class="comparison-table-wrapper">
        <table class="data-table" id="comparisonTable">
          <thead id="comparisonTableHead">
            <tr>
              <th>#</th>
              <th>Model / Attack</th>
              <th>Category</th>
              <th>Clean Acc.</th>
              <th>Degraded Acc.</th>
              <th>Evasion Rate</th>
              <th>Resilient Samples</th>
              <th>Mean Conf. Drop</th>
              <th>Avg Edits</th>
              <th>Avg Flip Budget</th>
            </tr>
          </thead>
          <tbody id="comparisonTableBody">
            <!-- Injected via JS -->
          </tbody>
        </table>
      </div>
    </section>

    <!-- Interactive Degradation Curve Chart Card -->
    <section class="chart-card">
      <div class="section-header" style="flex-wrap: wrap; gap: 12px;">
        <div>
          <h2 class="section-title" id="chartTitle">Degradation Curve Analysis</h2>
          <p class="kpi-subtext" id="chartSubtitle">Detection Rate (%) and Mean Malicious Confidence (%) vs Perturbation Budget (%)</p>
        </div>
        <div class="chart-toolbar">
          <select id="chartSeriesMode" class="filter-select" style="padding: 5px 10px; font-size: 12px;" onchange="renderChart()">
            <option value="auto">Curves: Auto (Best for Selection)</option>
            <option value="by_detector">Compare Detectors (Mean Across Selected Attacks)</option>
            <option value="by_category">Compare Attack Categories (Structure vs Content vs Metadata)</option>
            <option value="top_attacks">Top 6 Most Effective Attacks</option>
            <option value="active_attacks">All Attacks with Impact (&gt;0% Evasion/Drop)</option>
            <option value="all_runs">All Selected Runs</option>
          </select>

          <select id="chartMetricMode" class="filter-select" style="padding: 5px 10px; font-size: 12px;" onchange="renderChart()">
            <option value="detection">Y-Axis: Detection Rate (%)</option>
            <option value="evasion">Y-Axis: Evasion Rate (%)</option>
            <option value="confidence">Y-Axis: Mean Malicious Confidence (%)</option>
          </select>

          <div class="chart-view-toggle">
            <button class="chart-toggle-btn active" id="btnChartDynamic" onclick="setChartMode('interactive')">Interactive Plot</button>
            <button class="chart-toggle-btn" id="btnChartImage" onclick="setChartMode('image')">Saved Matplotlib PNG</button>
          </div>
        </div>
      </div>

      <div id="interactiveChartWrapper" class="canvas-container">
        <canvas id="chartCanvas"></canvas>
        <div id="chartTooltip" class="chart-tooltip"></div>
      </div>

      <div id="chartLegend" class="chart-legend"></div>

      <div id="imageChartWrapper" style="display: none; text-align: center; margin-top: 14px;">
        <img id="staticChartImg" src="" alt="Degradation Curve" style="max-width: 100%; border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
      </div>
    </section>

    <!-- Sample Trajectory Heatmap Grid -->
    <section class="timeline-card">
      <div class="section-header" style="flex-wrap: wrap; gap: 12px;">
        <div>
          <h2 class="section-title">Sample Evasion Progression Timeline</h2>
          <p class="kpi-subtext" id="timelineSubtitle">Interactive matrix of sample evasion status across budget steps. Click any cell to jump to its diff.</p>
        </div>
        <div class="timeline-toolbar">
          <div style="display: flex; align-items: center; gap: 6px;">
            <span style="font-size: 11px; color: var(--text-muted); font-weight: 600;">Run:</span>
            <select id="timelineRunSelect" class="filter-select" style="padding: 4px 10px; font-size: 12px;" onchange="onTimelineRunChange(this.value)">
              <!-- Populated via JS -->
            </select>
          </div>
          <div class="timeline-density-toggle">
            <span style="color:var(--text-dim); font-size:11px; margin: 0 4px;">Zoom:</span>
            <button class="density-btn" id="btnZoomCompact" onclick="setTimelineDensity('compact')">Compact</button>
            <button class="density-btn active" id="btnZoomNormal" onclick="setTimelineDensity('normal')">Normal</button>
            <button class="density-btn" id="btnZoomWide" onclick="setTimelineDensity('wide')">Wide</button>
          </div>
          <div style="display: flex; gap: 10px; font-size: 11px; align-items: center;">
            <span style="display: flex; align-items: center; gap: 4px;"><span class="timeline-cell cell-caught" style="width: 12px; height: 12px; min-width: 12px;"></span> Caught / Resisted</span>
            <span style="display: flex; align-items: center; gap: 4px;"><span class="timeline-cell cell-evaded" style="width: 12px; height: 12px; min-width: 12px;"></span> Evaded (Flipped)</span>
            <span style="display: flex; align-items: center; gap: 4px;"><span class="timeline-cell cell-failed" style="width: 12px; height: 12px; min-width: 12px;"></span> Failed (0 edits)</span>
          </div>
        </div>
      </div>

      <div class="timeline-scroll-container" id="timelineScrollContainer">
        <div class="timeline-table" id="timelineTable">
          <!-- Injected via JS with sticky budget header row & sticky sample column -->
        </div>
      </div>

      <div class="timeline-footer" id="timelineFooter">
        <div id="timelineStats">Showing samples</div>
        <div id="timelinePagination" style="display: flex; gap: 8px; align-items: center;"></div>
      </div>
    </section>

    <!-- Filters & Search Panel -->
    <section class="filter-panel" id="filterPanel">
      <div class="filter-row">
        <!-- Text Search -->
        <div class="search-box">
          <span class="search-icon">&#128269;</span>
          <input type="text" id="searchInput" class="search-input" placeholder="Search original or perturbed text, substitutions, detector, attack..." oninput="handleSearchInput()">
        </div>

        <!-- Outcome Filter -->
        <select id="filterOutcome" class="filter-select" onchange="applyFilters()">
          <option value="all">Outcome: All Results</option>
          <option value="evaded">Evaded Only (Attack Succeeded / Label Flipped)</option>
          <option value="caught">Caught Only (Detector Resisted)</option>
          <option value="failed">Failed Transformations</option>
        </select>

        <!-- Budget Step Filter -->
        <select id="filterBudget" class="filter-select" onchange="applyFilters()">
          <option value="all">Budget: All Steps</option>
          <!-- Injected via JS -->
        </select>

        <!-- Sample ID Filter -->
        <select id="filterSample" class="filter-select" onchange="applyFilters()">
          <option value="all">Sample: All Samples</option>
          <!-- Injected via JS -->
        </select>

        <!-- Min Confidence Drop -->
        <select id="filterMinDrop" class="filter-select" onchange="applyFilters()">
          <option value="0">Min Confidence Drop: Any</option>
          <option value="0.10">&ge; 10% Drop</option>
          <option value="0.25">&ge; 25% Drop</option>
          <option value="0.50">&ge; 50% Drop</option>
          <option value="0.75">&ge; 75% Drop</option>
        </select>

        <!-- Sort By -->
        <select id="sortSelect" class="filter-select" onchange="applyFilters()">
          <option value="evaded_first">Sort: Evaded &amp; Largest Drop First</option>
          <option value="drop_desc">Sort: Largest Confidence Drop (&darr;)</option>
          <option value="conf_asc">Sort: Lowest Perturbed Confidence (&uarr;)</option>
          <option value="edits_desc">Sort: Most Words Modified (&darr;)</option>
          <option value="default">Sort: Sample ID &amp; Budget</option>
        </select>

        <!-- Diff Mode Switcher -->
        <div class="diff-mode-switcher">
          <button class="diff-mode-btn active" id="btnDiffUnified" onclick="setGlobalDiffMode('unified')">Inline Diff</button>
          <button class="diff-mode-btn" id="btnDiffSplit" onclick="setGlobalDiffMode('split')">Side-by-Side</button>
          <button class="diff-mode-btn" id="btnDiffSubs" onclick="setGlobalDiffMode('subs')">Substitutions</button>
          <button class="diff-mode-btn" id="btnDiffPatch" onclick="setGlobalDiffMode('patch')">Git Patch</button>
        </div>
      </div>

      <div class="filter-stats-bar">
        <div>
          Showing <strong id="lblMatchesCount">0</strong> of <span id="lblTotalCount">0</span> iterations 
          (<strong id="lblEvadedCount">0</strong> label flips, <span id="lblResistedCount">0</span> resisted).
        </div>
        <div style="display: flex; gap: 10px;">
          <button class="btn" style="padding: 4px 8px; font-size: 11px;" onclick="resetFilters()">Reset Sample Filters</button>
        </div>
      </div>
    </section>

    <!-- Sample Cards Container -->
    <main class="samples-list" id="samplesList">
      <!-- Injected via JS -->
    </main>

    <!-- Pagination Bar -->
    <div class="pagination-bar" id="paginationBar">
      <!-- Injected via JS -->
    </div>

  </div>

  <!-- Raw Data Inspector Modal -->
  <div class="modal-overlay" id="rawModal" onclick="closeModal(event)">
    <div class="modal-dialog" onclick="event.stopPropagation()">
      <div class="modal-header">
        <h3 class="modal-title" id="modalTitle">Raw Iteration Data Inspector</h3>
        <button class="btn" style="padding: 4px 10px;" onclick="closeModal()">✕</button>
      </div>
      <div class="modal-body">
        <div style="display: flex; justify-content: flex-end; margin-bottom: 8px;">
          <button class="btn" onclick="copyModalContent()">Copy JSON</button>
        </div>
        <pre class="modal-code" id="modalCode"></pre>
      </div>
    </div>
  </div>

  <!-- Toast Notification -->
  <div id="toast" class="toast">Action completed</div>

  <!-- Embedded Report Data Payload -->
  <script>
    const BENCHMARK_DATA = __JSON_PAYLOAD__;
  </script>

  <!-- Client-Side Dashboard Application Script -->
  <script>
    // -------------------------------------------------------------------------
    // Global State & Indexed Metadata
    // -------------------------------------------------------------------------
    let selectedDetectorId = 'all'; // 'all' or specific detector_id
    let selectedAttackerId = 'all'; // 'all' or specific attacker_id
    let selectedTimelineRunKey = null; // key of run displayed in Sample Progression Timeline
    let globalDiffMode = 'unified'; // 'unified' | 'split' | 'subs' | 'patch'
    let chartViewMode = 'interactive'; // 'interactive' | 'image'
    let currentPage = 1;
    const pageSize = 15;

    let filteredRows = [];
    let uniqueDetectors = []; // [{ id, label }]
    let uniqueAttackers = []; // [{ id, label, category, meanEvasion, meanDrop }]
    let runGrid = {}; // `${att_id}__${det_id}` -> run object

    const DETECTOR_ORDER = ['logistic_regression', 'char_cnn', 'transformer'];
    const SERIES_PALETTE = [
      '#f43f5e', '#06b6d4', '#10b981', '#f59e0b', '#a855f7',
      '#3b82f6', '#ec4899', '#84cc16', '#14b8a6', '#eab308',
      '#6366f1', '#f97316', '#64748b'
    ];

    window.addEventListener('DOMContentLoaded', () => {
      buildDimensionIndices();
      initFilterDropdowns();
      initMatrixControls();
      renderInsightsBanner();
      if (BENCHMARK_DATA.length === 1) {
        selectedDetectorId = BENCHMARK_DATA[0].detector_id;
        selectedAttackerId = BENCHMARK_DATA[0].attacker_id;
      }
      selectSlice(selectedDetectorId, selectedAttackerId, false);
      setupCanvasResizing();
    });

    function showToast(msg) {
      const toast = document.getElementById('toast');
      toast.textContent = msg;
      toast.classList.add('show');
      setTimeout(() => toast.classList.remove('show'), 2200);
    }

    function catBadgeHtml(cat) {
      const c = (cat || 'content').toLowerCase();
      return `<span class="cat-badge cat-${c}">${escapeHtml(c)}</span>`;
    }

    // Build indices for Detectors, Attackers, and 2D Grid
    function buildDimensionIndices() {
      const detMap = new Map();
      const attMap = new Map();
      runGrid = {};

      BENCHMARK_DATA.forEach(d => {
        const detId = d.detector_id || d.key;
        const detLabel = d.detector_label || d.name;
        const attId = d.attacker_id || 'textfooler';
        const attLabel = d.attacker_label || 'TextFooler';
        const attCat = d.attack_category || 'content';

        d.detector_id = detId;
        d.detector_label = detLabel;
        d.attacker_id = attId;
        d.attacker_label = attLabel;
        d.attack_category = attCat;

        detMap.set(detId, detLabel);
        if (!attMap.has(attId)) {
          attMap.set(attId, { id: attId, label: attLabel, category: attCat, runs: [] });
        }
        attMap.get(attId).runs.push(d);
        runGrid[`${attId}__${detId}`] = d;
      });

      uniqueDetectors = Array.from(detMap.entries())
        .map(([id, label]) => {
          const runs = BENCHMARK_DATA.filter(r => r.detector_id === id);
          const avgDet = runs.reduce((acc, r) => acc + r.summary.final_detection_rate, 0) / (runs.length || 1);
          const avgEv = runs.reduce((acc, r) => acc + r.summary.final_evasion_rate, 0) / (runs.length || 1);
          const avgDrop = runs.reduce((acc, r) => acc + r.summary.mean_confidence_drop, 0) / (runs.length || 1);
          return { id, label, avgDet, avgEv, avgDrop, runs };
        })
        .sort((a, b) => {
          const ia = DETECTOR_ORDER.indexOf(a.id);
          const ib = DETECTOR_ORDER.indexOf(b.id);
          return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib) || a.label.localeCompare(b.label);
        });

      uniqueAttackers = Array.from(attMap.values())
        .map(a => {
          const meanEvasion = a.runs.reduce((acc, r) => acc + r.summary.final_evasion_rate, 0) / (a.runs.length || 1);
          const meanDrop = a.runs.reduce((acc, r) => acc + r.summary.mean_confidence_drop, 0) / (a.runs.length || 1);
          const maxEvasion = Math.max(...a.runs.map(r => r.summary.final_evasion_rate), 0);
          return { ...a, meanEvasion, meanDrop, maxEvasion };
        })
        .sort((a, b) => b.meanEvasion - a.meanEvasion || b.meanDrop - a.meanDrop || a.label.localeCompare(b.label));
    }

    function getActiveRuns() {
      return BENCHMARK_DATA.filter(d => {
        if (selectedDetectorId !== 'all' && d.detector_id !== selectedDetectorId) return false;
        if (selectedAttackerId !== 'all' && d.attacker_id !== selectedAttackerId) return false;
        return true;
      });
    }

    // -------------------------------------------------------------------------
    // Executive Insights Banner
    // -------------------------------------------------------------------------
    function renderInsightsBanner() {
      const banner = document.getElementById('insightsBanner');
      if (!banner) return;
      if (BENCHMARK_DATA.length <= 1) {
        banner.style.display = 'none';
        return;
      }
      banner.style.display = 'grid';
      banner.innerHTML = '';

      // 1. Most Effective Attack (by Mean Evasion)
      const topAttack = uniqueAttackers[0];
      const bestRunForTop = [...topAttack.runs].sort((a, b) => b.summary.final_evasion_rate - a.summary.final_evasion_rate)[0];

      // 2. Strongest Confidence Eroder
      const topDropAttack = [...uniqueAttackers].sort((a, b) => b.meanDrop - a.meanDrop)[0];
      const bestDropRun = [...topDropAttack.runs].sort((a, b) => b.summary.mean_confidence_drop - a.summary.mean_confidence_drop)[0];

      // 3. Detector Vulnerability Comparison
      const sortedDetsByVuln = [...uniqueDetectors].sort((a, b) => b.avgEv - a.avgEv || b.avgDrop - a.avgDrop);
      const mostVulnDet = sortedDetsByVuln[0];
      const mostResDet = sortedDetsByVuln[sortedDetsByVuln.length - 1];

      // 4. Category Breakdown
      const categories = ['structure', 'content', 'metadata'];
      const catStats = categories.map(cat => {
        const catRuns = BENCHMARK_DATA.filter(r => r.attack_category === cat);
        const avgEv = catRuns.length ? (catRuns.reduce((s, r) => s + r.summary.final_evasion_rate, 0) / catRuns.length) : 0;
        const avgDrp = catRuns.length ? (catRuns.reduce((s, r) => s + r.summary.mean_confidence_drop, 0) / catRuns.length) : 0;
        return { cat, avgEv, avgDrp, count: catRuns.length };
      }).filter(c => c.count > 0).sort((a, b) => b.avgEv - a.avgEv);

      const topCat = catStats[0];

      const insightCards = [
        {
          tag: 'Most Effective Attack (Label Flips)',
          badge: catBadgeHtml(topAttack.category),
          headline: `${escapeHtml(topAttack.label)} — ${topAttack.meanEvasion.toFixed(1)}% Mean Evasion`,
          detail: `Peaks at <strong>${bestRunForTop.summary.final_evasion_rate.toFixed(1)}%</strong> evasion against <strong>${escapeHtml(bestRunForTop.detector_label)}</strong> (-${bestRunForTop.summary.mean_confidence_drop.toFixed(1)}% &Delta;conf). Click to inspect.`,
          onClick: () => selectSlice('all', topAttack.id, true)
        },
        {
          tag: 'Strongest Confidence Erosion (&Delta;Conf)',
          badge: catBadgeHtml(topDropAttack.category),
          headline: `${escapeHtml(topDropAttack.label)} — -${topDropAttack.meanDrop.toFixed(1)}% Mean Drop`,
          detail: `Causes up to <strong>-${bestDropRun.summary.mean_confidence_drop.toFixed(1)}%</strong> confidence drop on <strong>${escapeHtml(bestDropRun.detector_label)}</strong>. Click to inspect.`,
          onClick: () => selectSlice('all', topDropAttack.id, true)
        },
        {
          tag: 'Detector Vulnerability Spread',
          badge: `<span class="meta-tag" style="padding:1px 6px; font-size:10px;">${uniqueDetectors.length} Detectors</span>`,
          headline: `${escapeHtml(mostVulnDet.label.split(' ')[0])} (${mostVulnDet.avgEv.toFixed(1)}% ev) vs ${escapeHtml(mostResDet.label.split(' ')[0])} (${mostResDet.avgEv.toFixed(1)}% ev)`,
          detail: `Most vulnerable: <strong>${escapeHtml(mostVulnDet.label)}</strong> (${mostVulnDet.avgDet.toFixed(1)}% avg retained). Most resilient: <strong>${escapeHtml(mostResDet.label)}</strong> (${mostResDet.avgDet.toFixed(1)}% avg retained).`,
          onClick: () => selectSlice(mostVulnDet.id, 'all', true)
        },
        {
          tag: 'Attack Vector Taxonomy Impact',
          badge: catBadgeHtml(topCat.cat),
          headline: `${topCat.cat.toUpperCase()} Attacks Lead (${topCat.avgEv.toFixed(1)}% Avg Evasion)`,
          detail: catStats.map(c => `<strong>${c.cat.charAt(0).toUpperCase() + c.cat.slice(1)}</strong>: ${c.avgEv.toFixed(1)}% ev (-${c.avgDrp.toFixed(1)}% &Delta;conf)`).join(' &bull; '),
          onClick: () => {
            const sel = document.getElementById('matrixCategorySelect');
            if (sel) { sel.value = topCat.cat; renderMatrix(); }
          }
        }
      ];

      insightCards.forEach(ic => {
        const card = document.createElement('div');
        card.className = 'insight-card';
        card.innerHTML = `
          <div class="insight-tag"><span>${ic.tag}</span>${ic.badge}</div>
          <div class="insight-headline">${ic.headline}</div>
          <div class="insight-detail">${ic.detail}</div>
        `;
        card.onclick = ic.onClick;
        banner.appendChild(card);
      });
    }

    // -------------------------------------------------------------------------
    // 2D Attack x Detector Heatmap Matrix
    // -------------------------------------------------------------------------
    function initMatrixControls() {
      const bSelect = document.getElementById('matrixBudgetSelect');
      if (!bSelect || BENCHMARK_DATA.length === 0) return;
      const budgets = BENCHMARK_DATA[0].metrics.map(m => m.budget);
      const maxPct = Math.round((budgets[budgets.length - 1] || 0.2) * 100);
      bSelect.innerHTML = `<option value="max">Budget: Max (${maxPct}%)</option>`;
      budgets.forEach((b, idx) => {
        const pct = Math.round(b * 100);
        const opt = document.createElement('option');
        opt.value = String(idx);
        opt.textContent = `Budget Step ${idx + 1}: ${pct}% (${b.toFixed(2)})`;
        bSelect.appendChild(opt);
      });
    }

    function getRunMetricsAtBudget(run, budgetSel) {
      if (!run) return null;
      if (budgetSel === 'max' || !run.metrics || run.metrics.length === 0) {
        return {
          evasion: run.summary.final_evasion_rate,
          detection: run.summary.final_detection_rate,
          drop: run.summary.mean_confidence_drop,
          flipBudget: run.summary.avg_flip_budget
        };
      }
      const idx = Math.min(parseInt(budgetSel, 10) || 0, run.metrics.length - 1);
      const m = run.metrics[idx];
      return {
        evasion: Number(((1.0 - m.detection_rate) * 100).toFixed(1)),
        detection: Number((m.detection_rate * 100).toFixed(1)),
        drop: Number((m.mean_confidence_drop * 100).toFixed(2)),
        flipBudget: run.summary.avg_flip_budget
      };
    }

    function computeHeatColor(metricMode, stats) {
      if (!stats) return 'rgba(15, 23, 42, 0.5)';
      if (metricMode === 'detection') {
        // High detection = green, low detection = red
        const evRatio = Math.min(1, Math.max(0, (100 - stats.detection) / 100));
        if (evRatio < 0.05) return 'rgba(16, 185, 129, 0.14)';
        return `rgba(244, 63, 94, ${(0.15 + evRatio * 0.55).toFixed(2)})`;
      }
      if (metricMode === 'drop') {
        const ratio = Math.min(1, Math.max(0, stats.drop / 60));
        if (stats.drop <= 0.1) return 'rgba(16, 185, 129, 0.10)';
        return `rgba(244, 63, 94, ${(0.12 + ratio * 0.58).toFixed(2)})`;
      }
      // Default ('combined', 'evasion', 'flip'): color by evasion + subtle amber for confidence drop
      const ev = stats.evasion;
      const drp = stats.drop;
      if (ev >= 50) {
        const alpha = 0.32 + Math.min(0.45, ((ev - 50) / 50) * 0.40);
        return `rgba(244, 63, 94, ${alpha.toFixed(2)})`;
      } else if (ev > 0) {
        const alpha = 0.18 + (ev / 50) * 0.22;
        return `rgba(245, 158, 11, ${alpha.toFixed(2)})`;
      } else if (drp >= 5.0) {
        // 0% evasion but noticeable confidence erosion
        return 'rgba(56, 189, 248, 0.14)';
      } else {
        return 'rgba(16, 185, 129, 0.10)';
      }
    }

    function formatMatrixCellHtml(metricMode, stats) {
      if (!stats) return `<div class="heat-main-val" style="color:var(--text-dim);">—</div>`;
      if (metricMode === 'evasion') {
        return `
          <div class="heat-main-val" style="color:${stats.evasion > 0 ? '#fff' : 'var(--text-muted)'}">${stats.evasion.toFixed(1)}%</div>
          <div class="heat-sub-val">Evaded</div>
        `;
      }
      if (metricMode === 'detection') {
        return `
          <div class="heat-main-val">${stats.detection.toFixed(1)}%</div>
          <div class="heat-sub-val">Retained</div>
        `;
      }
      if (metricMode === 'drop') {
        const sign = stats.drop > 0 ? '-' : '';
        return `
          <div class="heat-main-val" style="color:${stats.drop >= 5 ? '#fda4af' : 'var(--text-main)'}">${sign}${stats.drop.toFixed(1)}%</div>
          <div class="heat-sub-val">&Delta;Conf (${stats.evasion.toFixed(0)}% ev)</div>
        `;
      }
      if (metricMode === 'flip') {
        const fb = stats.flipBudget !== null && stats.flipBudget !== undefined ? `${stats.flipBudget}%` : 'No Flips';
        return `
          <div class="heat-main-val">${fb}</div>
          <div class="heat-sub-val">${stats.evasion.toFixed(1)}% Evaded</div>
        `;
      }
      // 'combined' mode: Evasion % primary + ΔConf % secondary
      const dropStr = stats.drop > 0 ? `-${stats.drop.toFixed(1)}% &Delta;conf` : `0.0% &Delta;conf`;
      const mainColor = stats.evasion >= 50 ? '#fda4af' : (stats.evasion > 0 ? '#fde68a' : (stats.drop >= 5 ? '#7dd3fc' : '#6ee7b7'));
      return `
        <div class="heat-main-val" style="color:${mainColor};">${stats.evasion.toFixed(1)}% <span style="font-size:11px; font-weight:600; opacity:0.85;">ev</span></div>
        <div class="heat-sub-val">${dropStr}</div>
      `;
    }

    function renderMatrix() {
      const sec = document.getElementById('matrixSection');
      if (!sec) return;
      if (uniqueAttackers.length <= 1 && uniqueDetectors.length <= 1) {
        sec.style.display = 'none';
        return;
      }
      sec.style.display = 'block';

      const metricMode = document.getElementById('matrixMetricSelect')?.value || 'combined';
      const budgetSel = document.getElementById('matrixBudgetSelect')?.value || 'max';
      const catFilter = document.getElementById('matrixCategorySelect')?.value || 'all';
      const sortMode = document.getElementById('matrixSortSelect')?.value || 'evasion_desc';

      const thead = document.getElementById('matrixThead');
      const tbody = document.getElementById('matrixTbody');
      const tfoot = document.getElementById('matrixTfoot');

      // 1. Build Header
      let headHtml = `<tr>
        <th style="width: 44px;" onclick="selectSlice('all', 'all', true)" title="Click to reset selection">#</th>
        <th onclick="selectSlice('all', 'all', true)" title="Click to view all attacks">Attack Algorithm</th>
        <th style="width: 110px;">Category</th>`;

      uniqueDetectors.forEach(det => {
        const isDetSelected = selectedDetectorId === det.id && selectedAttackerId === 'all';
        const activeStyle = isDetSelected ? 'border-bottom: 2px solid var(--accent-cyan); color: #fff;' : '';
        headHtml += `<th style="${activeStyle}" onclick="selectSlice('${det.id}', 'all', true)" title="Click to filter dashboard by ${escapeHtml(det.label)}">
          ${escapeHtml(det.label)}
          <div style="font-size:10px; font-weight:500; color:var(--accent-cyan); text-transform:none; margin-top:2px;">Click to slice column &darr;</div>
        </th>`;
      });

      headHtml += `<th style="background: rgba(99, 102, 241, 0.12); color: #c7d2fe;" onclick="selectSlice('all', 'all', true)">
        Mean Across Detectors
        <div style="font-size:10px; font-weight:500; color:var(--text-muted); text-transform:none; margin-top:2px;">Overall Effectiveness</div>
      </th></tr>`;
      thead.innerHTML = headHtml;

      // 2. Compute rows for each attacker at selected budget
      let rowsData = uniqueAttackers
        .filter(att => catFilter === 'all' || att.category === catFilter)
        .map(att => {
          const cells = uniqueDetectors.map(det => {
            const run = runGrid[`${att.id}__${det.id}`];
            return { det, run, stats: getRunMetricsAtBudget(run, budgetSel) };
          });
          const validStats = cells.map(c => c.stats).filter(Boolean);
          const meanEv = validStats.length ? validStats.reduce((s, x) => s + x.evasion, 0) / validStats.length : 0;
          const meanDet = validStats.length ? validStats.reduce((s, x) => s + x.detection, 0) / validStats.length : 100;
          const meanDrop = validStats.length ? validStats.reduce((s, x) => s + x.drop, 0) / validStats.length : 0;
          const validFlips = validStats.map(x => x.flipBudget).filter(x => x !== null && x !== undefined);
          const meanFlip = validFlips.length ? Number((validFlips.reduce((s, x) => s + x, 0) / validFlips.length).toFixed(1)) : null;

          return {
            att,
            cells,
            meanStats: { evasion: meanEv, detection: meanDet, drop: meanDrop, flipBudget: meanFlip }
          };
        });

      // Sort rows
      if (sortMode === 'evasion_desc') {
        rowsData.sort((a, b) => b.meanStats.evasion - a.meanStats.evasion || b.meanStats.drop - a.meanStats.drop);
      } else if (sortMode === 'drop_desc') {
        rowsData.sort((a, b) => b.meanStats.drop - a.meanStats.drop || b.meanStats.evasion - a.meanStats.evasion);
      } else if (sortMode === 'category') {
        rowsData.sort((a, b) => a.att.category.localeCompare(b.att.category) || b.meanStats.evasion - a.meanStats.evasion);
      } else if (sortMode === 'name_asc') {
        rowsData.sort((a, b) => a.att.label.localeCompare(b.att.label));
      }

      // 3. Render Body Rows
      tbody.innerHTML = '';
      rowsData.forEach((rowObj, idx) => {
        const tr = document.createElement('tr');
        const isRowSelected = selectedAttackerId === rowObj.att.id && selectedDetectorId === 'all';
        if (isRowSelected) {
          tr.style.outline = '1px solid var(--accent-cyan)';
        }

        const rankNum = idx + 1;
        const isTop = rowObj.meanStats.evasion > 0 && rankNum <= 3;

        let trHtml = `
          <td><span class="matrix-rank-badge ${isTop ? 'top-rank' : ''}">#${rankNum}</span></td>
          <td class="matrix-attack-cell" onclick="selectSlice('all', '${rowObj.att.id}', true)" title="Click to compare all detectors under ${escapeHtml(rowObj.att.label)}">
            ${escapeHtml(rowObj.att.label)}
          </td>
          <td>${catBadgeHtml(rowObj.att.category)}</td>
        `;

        rowObj.cells.forEach(c => {
          const bg = computeHeatColor(metricMode, c.stats);
          const isSelectedCell = (selectedDetectorId === c.det.id && selectedAttackerId === rowObj.att.id);
          const cellClass = 'matrix-heat-cell' + (isSelectedCell ? ' selected-cell' : '');
          const cellTitle = c.stats
            ? `${rowObj.att.label} vs ${c.det.label}\nEvasion: ${c.stats.evasion.toFixed(1)}% | Retained: ${c.stats.detection.toFixed(1)}%\nMean Conf Drop: -${c.stats.drop.toFixed(2)}%\nClick to inspect this run`
            : 'Run not available';
          trHtml += `<td class="${cellClass}" style="background:${bg};" title="${escapeHtml(cellTitle)}" onclick="selectSlice('${c.det.id}', '${rowObj.att.id}', true)">
            ${formatMatrixCellHtml(metricMode, c.stats)}
          </td>`;
        });

        const meanBg = computeHeatColor(metricMode, rowObj.meanStats);
        trHtml += `<td class="matrix-heat-cell" style="background:${meanBg}; border-left: 2px solid var(--border-accent);" title="Click to compare all detectors for ${escapeHtml(rowObj.att.label)}" onclick="selectSlice('all', '${rowObj.att.id}', true)">
          ${formatMatrixCellHtml(metricMode, rowObj.meanStats)}
        </td>`;

        tr.innerHTML = trHtml;
        tbody.appendChild(tr);
      });

      // 4. Render Footer Row (Detector Averages across visible rows)
      if (rowsData.length > 0) {
        let footHtml = `<tr class="matrix-footer-row">
          <td></td>
          <td style="color:#fff;">DETECTOR AVERAGE (${rowsData.length} Attacks)</td>
          <td><span class="cat-badge" style="background:rgba(255,255,255,0.08); color:#cbd5e1;">MEAN</span></td>`;

        uniqueDetectors.forEach((det, dIdx) => {
          const colStats = rowsData.map(r => r.cells[dIdx].stats).filter(Boolean);
          const avgEv = colStats.length ? colStats.reduce((s, x) => s + x.evasion, 0) / colStats.length : 0;
          const avgDet = colStats.length ? colStats.reduce((s, x) => s + x.detection, 0) / colStats.length : 100;
          const avgDrop = colStats.length ? colStats.reduce((s, x) => s + x.drop, 0) / colStats.length : 0;
          const validFlips = colStats.map(x => x.flipBudget).filter(x => x !== null && x !== undefined);
          const avgFlip = validFlips.length ? Number((validFlips.reduce((s, x) => s + x, 0) / validFlips.length).toFixed(1)) : null;
          const aggStats = { evasion: avgEv, detection: avgDet, drop: avgDrop, flipBudget: avgFlip };

          footHtml += `<td class="matrix-heat-cell" onclick="selectSlice('${det.id}', 'all', true)" title="Click to view all attacks against ${escapeHtml(det.label)}">
            ${formatMatrixCellHtml(metricMode, aggStats)}
          </td>`;
        });

        const allMeans = rowsData.map(r => r.meanStats);
        const grandEv = allMeans.reduce((s, x) => s + x.evasion, 0) / allMeans.length;
        const grandDet = allMeans.reduce((s, x) => s + x.detection, 0) / allMeans.length;
        const grandDrop = allMeans.reduce((s, x) => s + x.drop, 0) / allMeans.length;
        const grandFlips = allMeans.map(x => x.flipBudget).filter(x => x !== null && x !== undefined);
        const grandFlip = grandFlips.length ? Number((grandFlips.reduce((s, x) => s + x, 0) / grandFlips.length).toFixed(1)) : null;

        footHtml += `<td class="matrix-heat-cell" style="border-left: 2px solid var(--border-accent);" onclick="selectSlice('all', 'all', true)">
          ${formatMatrixCellHtml(metricMode, { evasion: grandEv, detection: grandDet, drop: grandDrop, flipBudget: grandFlip })}
        </td></tr>`;
        tfoot.innerHTML = footHtml;
      } else {
        tfoot.innerHTML = '';
      }
    }

    // -------------------------------------------------------------------------
    // Two-Level Dimensional Navigation Bar (Detector + Attack Selector)
    // -------------------------------------------------------------------------
    function renderDimensionNav() {
      const detPills = document.getElementById('detectorPills');
      const attPills = document.getElementById('attackerPills');
      const sliceText = document.getElementById('activeSliceText');
      const resetBtn = document.getElementById('btnResetSlice');
      if (!detPills || !attPills) return;

      detPills.innerHTML = '';
      attPills.innerHTML = '';

      // 1. Detector Pills
      if (uniqueDetectors.length > 1) {
        const allDetBtn = document.createElement('button');
        allDetBtn.className = 'detector-tab' + (selectedDetectorId === 'all' ? ' active' : '');
        allDetBtn.innerHTML = `All Detectors <span class="detector-tab-badge">${uniqueDetectors.length}</span>`;
        allDetBtn.onclick = () => selectSlice('all', selectedAttackerId, false);
        detPills.appendChild(allDetBtn);
      }

      uniqueDetectors.forEach(det => {
        const btn = document.createElement('button');
        btn.className = 'detector-tab' + (selectedDetectorId === det.id ? ' active' : '');
        btn.innerHTML = `${escapeHtml(det.label)} <span class="detector-tab-badge">${det.avgEv.toFixed(1)}% avg ev</span>`;
        btn.onclick = () => selectSlice(det.id, selectedAttackerId, false);
        detPills.appendChild(btn);
      });

      // 2. Attacker Pills (Sorted by effectiveness)
      if (uniqueAttackers.length > 1) {
        const allAttBtn = document.createElement('button');
        allAttBtn.className = 'detector-tab' + (selectedAttackerId === 'all' ? ' active' : '');
        allAttBtn.innerHTML = `All Attacks <span class="detector-tab-badge">${uniqueAttackers.length}</span>`;
        allAttBtn.onclick = () => selectSlice(selectedDetectorId, 'all', false);
        attPills.appendChild(allAttBtn);
      }

      uniqueAttackers.forEach(att => {
        const btn = document.createElement('button');
        btn.className = 'detector-tab' + (selectedAttackerId === att.id ? ' active' : '');
        // If a specific detector is selected, show that detector's evasion rate on the pill badge
        let badgeVal = `${att.meanEvasion.toFixed(0)}%`;
        if (selectedDetectorId !== 'all') {
          const r = runGrid[`${att.id}__${selectedDetectorId}`];
          if (r) badgeVal = `${r.summary.final_evasion_rate.toFixed(0)}%`;
        }
        btn.innerHTML = `${escapeHtml(att.label)} <span class="detector-tab-badge">${badgeVal}</span>`;
        btn.onclick = () => selectSlice(selectedDetectorId, att.id, false);
        attPills.appendChild(btn);
      });

      // 3. Active Slice Status Bar
      const activeRuns = getActiveRuns();
      const detTitle = selectedDetectorId === 'all'
        ? `All Detectors (${uniqueDetectors.length})`
        : (uniqueDetectors.find(d => d.id === selectedDetectorId)?.label || selectedDetectorId);
      const attTitle = selectedAttackerId === 'all'
        ? `All Attacks (${uniqueAttackers.length})`
        : (uniqueAttackers.find(a => a.id === selectedAttackerId)?.label || selectedAttackerId);

      if (sliceText) {
        sliceText.innerHTML = `Active Slice: <strong style="color:#fff;">${escapeHtml(detTitle)}</strong> &times; <strong style="color:var(--accent-cyan);">${escapeHtml(attTitle)}</strong> <span style="margin-left:6px; opacity:0.8;">(${activeRuns.length} run${activeRuns.length === 1 ? '' : 's'} matched)</span>`;
      }
      if (resetBtn) {
        resetBtn.style.display = (selectedDetectorId !== 'all' || selectedAttackerId !== 'all') && BENCHMARK_DATA.length > 1 ? 'inline-flex' : 'none';
      }
    }

    function selectSlice(detId, attId, scrollToDetails = false) {
      selectedDetectorId = detId;
      selectedAttackerId = attId;

      renderMatrix();
      renderDimensionNav();
      renderKPICards();
      renderComparisonTable();
      populateTimelineRunSelector();
      renderTimelineGrid();
      updateStaticChartImage();
      renderChart();
      applyFilters();

      if (scrollToDetails) {
        const navEl = document.getElementById('detectorNav');
        if (navEl) navEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }

    // Legacy compatibility helper
    function switchDetector(idx) {
      if (idx === -1) {
        selectSlice('all', 'all', false);
      } else if (BENCHMARK_DATA[idx]) {
        selectSlice(BENCHMARK_DATA[idx].detector_id, BENCHMARK_DATA[idx].attacker_id, false);
      }
    }

    // -------------------------------------------------------------------------
    // KPI Summary Cards (Context-Aware for Active Slice)
    // -------------------------------------------------------------------------
    function renderKPICards() {
      const grid = document.getElementById('kpiGrid');
      grid.innerHTML = '';

      const runs = getActiveRuns();
      if (runs.length === 0) return;

      const totalSamples = runs.reduce((acc, d) => acc + d.summary.total_samples, 0);
      const totalEvaded = runs.reduce((acc, d) => acc + d.summary.total_evaded_samples, 0);
      const sumEvasion = runs.reduce((acc, d) => acc + d.summary.final_evasion_rate, 0);
      const overallEvasion = (sumEvasion / runs.length).toFixed(1);
      const sumDrop = runs.reduce((acc, d) => acc + d.summary.mean_confidence_drop, 0);
      const meanDrop = (sumDrop / runs.length).toFixed(1);
      const sumEdits = runs.reduce((acc, d) => acc + d.summary.avg_budget_consumed, 0);
      const avgEdits = (sumEdits / runs.length).toFixed(1);
      const failedTransforms = runs.reduce((acc, d) => acc + d.summary.total_failed_transforms, 0);

      const cards = [
        {
          label: runs.length > 1 ? 'Mean Evasion Rate (Slice)' : 'Evasion Rate (Attack Success)',
          value: `${overallEvasion}%`,
          subtext: `${totalEvaded} of ${totalSamples} sample evaluations evaded`,
          accent: 'var(--evaded-red)'
        },
        {
          label: 'Detector Resiliency (Retained)',
          value: `${(100 - parseFloat(overallEvasion)).toFixed(1)}%`,
          subtext: `${totalSamples - totalEvaded} sample evaluations remained caught`,
          accent: 'var(--caught-green)'
        },
        {
          label: 'Mean Confidence Drop',
          value: `-${meanDrop}%`,
          subtext: 'Average malicious certainty erosion at max budget',
          accent: 'var(--accent-primary)'
        },
        {
          label: 'Avg Perturbation Edits',
          value: `${avgEdits}`,
          subtext: 'Modified tokens/injections at max budget',
          accent: 'var(--accent-cyan)'
        },
        {
          label: 'Failed Transformations',
          value: `${failedTransforms}`,
          subtext: 'Budget steps where 0 edits were applicable',
          accent: 'var(--text-dim)'
        }
      ];

      cards.forEach(c => {
        const div = document.createElement('div');
        div.className = 'kpi-card';
        div.style.setProperty('--kpi-accent', c.accent);
        div.innerHTML = `
          <div class="kpi-label">${c.label}</div>
          <div class="kpi-value">${c.value}</div>
          <div class="kpi-subtext">${c.subtext}</div>
        `;
        grid.appendChild(div);
      });
    }

    // -------------------------------------------------------------------------
    // Context-Aware Comparison / Leaderboard Table
    // -------------------------------------------------------------------------
    function renderComparisonTable() {
      const sec = document.getElementById('comparisonSection');
      if (BENCHMARK_DATA.length <= 1) {
        sec.style.display = 'none';
        return;
      }
      sec.style.display = 'block';

      const groupSelect = document.getElementById('comparisonGroupSelect')?.value || 'auto';
      const titleEl = document.getElementById('comparisonTitle');
      const subEl = document.getElementById('comparisonSubtitle');
      const thead = document.getElementById('comparisonTableHead');
      const tbody = document.getElementById('comparisonTableBody');
      tbody.innerHTML = '';

      // Determine effective grouping mode
      let mode = groupSelect;
      if (mode === 'auto') {
        if (selectedAttackerId !== 'all' && selectedDetectorId === 'all') {
          mode = 'single_attack_detectors';
        } else if (selectedDetectorId !== 'all' && selectedAttackerId === 'all') {
          mode = 'single_detector_attacks';
        } else if (selectedDetectorId !== 'all' && selectedAttackerId !== 'all') {
          mode = 'single_attack_detectors'; // Show all detectors for this attack so user can compare
        } else {
          mode = 'by_attack';
        }
      }

      if (mode === 'by_attack') {
        const attObjTitle = 'Attack Effectiveness Leaderboard (Averaged Across Detectors)';
        if (titleEl) titleEl.textContent = attObjTitle;
        if (subEl) subEl.textContent = 'Attacks ranked from highest to lowest Mean Evasion Rate (%) and Mean Confidence Drop (-Δ%). Click any row to focus on that attack.';
        thead.innerHTML = `<tr>
          <th>Rank</th>
          <th>Attack Algorithm</th>
          <th>Category</th>
          <th>Mean Degraded Acc.</th>
          <th>Mean Evasion Rate</th>
          <th>Peak Evasion (Hardest Hit Detector)</th>
          <th>Mean Conf. Drop</th>
          <th>Avg Edits</th>
        </tr>`;

        uniqueAttackers.forEach((att, idx) => {
          const runs = att.runs;
          const meanDet = (runs.reduce((s, r) => s + r.summary.final_detection_rate, 0) / runs.length).toFixed(1);
          const meanEv = att.meanEvasion.toFixed(1);
          const meanDrp = att.meanDrop.toFixed(2);
          const avgEd = (runs.reduce((s, r) => s + r.summary.avg_budget_consumed, 0) / runs.length).toFixed(1);
          const bestRun = [...runs].sort((a, b) => b.summary.final_evasion_rate - a.summary.final_evasion_rate || b.summary.mean_confidence_drop - a.summary.mean_confidence_drop)[0];

          const tr = document.createElement('tr');
          tr.style.cursor = 'pointer';
          if (selectedAttackerId === att.id) tr.style.background = 'rgba(99, 102, 241, 0.10)';
          tr.innerHTML = `
            <td><span class="matrix-rank-badge ${att.meanEvasion > 0 && idx < 3 ? 'top-rank' : ''}">#${idx + 1}</span></td>
            <td><strong>${escapeHtml(att.label)}</strong></td>
            <td>${catBadgeHtml(att.category)}</td>
            <td><span style="color:${ parseFloat(meanDet) < 50 ? 'var(--evaded-red)' : 'var(--caught-green)'}; font-weight:600;">${meanDet}%</span></td>
            <td><span class="status-badge ${att.meanEvasion >= 50 ? 'status-evaded' : (att.meanEvasion > 0 ? 'budget-badge' : 'status-caught')}">${meanEv}%</span></td>
            <td><strong>${bestRun.summary.final_evasion_rate.toFixed(1)}%</strong> on <span style="color:var(--text-muted);">${escapeHtml(bestRun.detector_label)}</span></td>
            <td>-${meanDrp}%</td>
            <td>${avgEd} words</td>
          `;
          tr.onclick = () => selectSlice('all', att.id, false);
          tbody.appendChild(tr);
        });
        return;
      }

      if (mode === 'by_detector') {
        if (titleEl) titleEl.textContent = 'Detector Robustness Summary (Averaged Across All Attacks)';
        if (subEl) subEl.textContent = 'Detectors ranked from most vulnerable to most resilient across all evaluated attack algorithms.';
        thead.innerHTML = `<tr>
          <th>Rank</th>
          <th>Detector Model</th>
          <th>Clean Acc.</th>
          <th>Avg Degraded Acc.</th>
          <th>Avg Evasion Rate</th>
          <th>Avg Conf. Drop</th>
          <th>Most Effective Attack Against Model</th>
        </tr>`;

        const sortedDets = [...uniqueDetectors].sort((a, b) => b.avgEv - a.avgEv || b.avgDrop - a.avgDrop);
        sortedDets.forEach((det, idx) => {
          const worstRun = [...det.runs].sort((a, b) => b.summary.final_evasion_rate - a.summary.final_evasion_rate || b.summary.mean_confidence_drop - a.summary.mean_confidence_drop)[0];
          const tr = document.createElement('tr');
          tr.style.cursor = 'pointer';
          tr.innerHTML = `
            <td><span class="matrix-rank-badge">#${idx + 1}</span></td>
            <td><strong>${escapeHtml(det.label)}</strong></td>
            <td><span style="color:var(--caught-green); font-weight:600;">100.0%</span></td>
            <td><span style="color:${det.avgDet < 50 ? 'var(--evaded-red)' : 'var(--caught-green)'}; font-weight:600;">${det.avgDet.toFixed(1)}%</span></td>
            <td><span class="status-badge ${det.avgEv >= 50 ? 'status-evaded' : 'status-caught'}">${det.avgEv.toFixed(1)}%</span></td>
            <td>-${det.avgDrop.toFixed(2)}%</td>
            <td><strong>${escapeHtml(worstRun.attacker_label)}</strong> (${worstRun.summary.final_evasion_rate.toFixed(1)}% ev, -${worstRun.summary.mean_confidence_drop.toFixed(1)}% &Delta;conf)</td>
          `;
          tr.onclick = () => selectSlice(det.id, 'all', false);
          tbody.appendChild(tr);
        });
        return;
      }

      // Single Detector (rank all attacks) OR Single Attack (compare all detectors) OR All Runs
      let rowsToRender = [];
      if (mode === 'single_detector_attacks') {
        const detObj = uniqueDetectors.find(d => d.id === selectedDetectorId);
        if (titleEl) titleEl.textContent = `Attack Leaderboard Against ${detObj ? detObj.label : selectedDetectorId}`;
        if (subEl) subEl.textContent = 'All evaluated attacks ranked by Evasion Rate (%) and Confidence Drop (-Δ%) against this detector.';
        rowsToRender = BENCHMARK_DATA.filter(d => d.detector_id === selectedDetectorId);
      } else if (mode === 'single_attack_detectors') {
        const attObj = uniqueAttackers.find(a => a.id === selectedAttackerId);
        if (titleEl) titleEl.textContent = `Detector Comparison Under ${attObj ? attObj.label : selectedAttackerId}`;
        if (subEl) subEl.textContent = 'Side-by-side comparison of detector resistance against this specific attack algorithm.';
        rowsToRender = BENCHMARK_DATA.filter(d => d.attacker_id === selectedAttackerId);
      } else {
        if (titleEl) titleEl.textContent = `All Individual Benchmark Runs (${BENCHMARK_DATA.length} Runs Ranked by Evasion)`;
        if (subEl) subEl.textContent = 'Complete list of (Detector, Attack) combinations sorted from highest evasion impact to lowest.';
        rowsToRender = [... getActiveRuns()];
      }

      rowsToRender.sort((a, b) => b.summary.final_evasion_rate - a.summary.final_evasion_rate || b.summary.mean_confidence_drop - a.summary.mean_confidence_drop);

      thead.innerHTML = `<tr>
        <th>Rank</th>
        <th>Detector Model</th>
        <th>Attack Algorithm</th>
        <th>Category</th>
        <th>Degraded Acc.</th>
        <th>Evasion Rate</th>
        <th>Resilient Samples</th>
        <th>Mean Conf. Drop</th>
        <th>Avg Edits</th>
        <th>Avg Flip Budget</th>
      </tr>`;

      rowsToRender.forEach((det, idx) => {
        const tr = document.createElement('tr');
        const s = det.summary;
        const degAcc = `${s.final_detection_rate}%`;
        const evRate = `${s.final_evasion_rate}%`;
        const flipStr = s.avg_flip_budget ? `${s.avg_flip_budget}% budget` : 'N/A';
        const isCurrent = (selectedDetectorId === det.detector_id && selectedAttackerId === det.attacker_id);
        if (isCurrent) tr.style.background = 'rgba(6, 182, 212, 0.12)';

        tr.innerHTML = `
          <td><span class="matrix-rank-badge ${s.final_evasion_rate > 0 && idx < 3 ? 'top-rank' : ''}">#${idx + 1}</span></td>
          <td><strong>${escapeHtml(det.detector_label)}</strong></td>
          <td><strong>${escapeHtml(det.attacker_label)}</strong></td>
          <td>${catBadgeHtml(det.attack_category)}</td>
          <td><span style="color: ${s.final_detection_rate < 50 ? 'var(--evaded-red)' : 'var(--caught-green)'}; font-weight:600;">${degAcc}</span></td>
          <td><span class="status-badge ${s.final_evasion_rate >= 50 ? 'status-evaded' : (s.final_evasion_rate > 0 ? 'budget-badge' : 'status-caught')}">${evRate}</span></td>
          <td>${s.resilient_samples} / ${s.total_samples}</td>
          <td>-${s.mean_confidence_drop}%</td>
          <td>${s.avg_budget_consumed} words</td>
          <td>${flipStr}</td>
        `;
        tr.style.cursor = 'pointer';
        tr.onclick = () => selectSlice(det.detector_id, det.attacker_id, false);
        tbody.appendChild(tr);
      });
    }

    // -------------------------------------------------------------------------
    // Timeline Progression Grid State & Run Selector
    // -------------------------------------------------------------------------
    let timelineDensity = 'normal'; // 'compact' | 'normal' | 'wide'
    let timelinePage = 1;
    let timelinePageSize = 50;

    function populateTimelineRunSelector() {
      const sel = document.getElementById('timelineRunSelect');
      if (!sel) return;
      const activeRuns = [...getActiveRuns()].sort(
        (a, b) => b.summary.final_evasion_rate - a.summary.final_evasion_rate || b.summary.mean_confidence_drop - a.summary.mean_confidence_drop
      );
      sel.innerHTML = '';
      activeRuns.forEach(r => {
        const opt = document.createElement('option');
        opt.value = r.key;
        opt.textContent = `${r.detector_label} — ${r.attacker_label} (${r.summary.final_evasion_rate}% ev, -${r.summary.mean_confidence_drop}% Δconf)`;
        sel.appendChild(opt);
      });

      if (activeRuns.length > 0) {
        const stillValid = activeRuns.some(r => r.key === selectedTimelineRunKey);
        if (!stillValid) {
          selectedTimelineRunKey = activeRuns[0].key;
        }
        sel.value = selectedTimelineRunKey;
      }
    }

    function onTimelineRunChange(runKey) {
      selectedTimelineRunKey = runKey;
      renderTimelineGrid();
    }

    function getTimelineTargetRun() {
      const activeRuns = getActiveRuns();
      if (selectedTimelineRunKey) {
        const found = activeRuns.find(r => r.key === selectedTimelineRunKey) || BENCHMARK_DATA.find(r => r.key === selectedTimelineRunKey);
        if (found) return found;
      }
      return activeRuns[0] || BENCHMARK_DATA[0];
    }

    function setTimelineDensity(d) {
      timelineDensity = d;
      const bCompact = document.getElementById('btnZoomCompact');
      const bNormal = document.getElementById('btnZoomNormal');
      const bWide = document.getElementById('btnZoomWide');
      if (bCompact) bCompact.classList.toggle('active', d === 'compact');
      if (bNormal) bNormal.classList.toggle('active', d === 'normal');
      if (bWide) bWide.classList.toggle('active', d === 'wide');

      const w = d === 'compact' ? '14px' : (d === 'wide' ? '48px' : '32px');
      const container = document.getElementById('timelineScrollContainer');
      if (container) container.style.setProperty('--cell-w', w);
      renderTimelineGrid();
    }

    function changeTimelinePage(delta) {
      timelinePage += delta;
      renderTimelineGrid();
    }

    function setTimelinePageSize(sz) {
      timelinePageSize = sz === 'all' ? 999999 : parseInt(sz, 10);
      timelinePage = 1;
      renderTimelineGrid();
    }

    function renderTimelineGrid() {
      const container = document.getElementById('timelineTable');
      if (!container) return;
      container.innerHTML = '';

      const det = getTimelineTargetRun();
      if (!det) return;

      const subEl = document.getElementById('timelineSubtitle');
      if (subEl) {
        subEl.innerHTML = `Showing sample progression for <strong>${escapeHtml(det.detector_label)}</strong> under <strong>${escapeHtml(det.attacker_label)}</strong> (${det.summary.final_evasion_rate}% evasion). Click any cell to jump to its diff below.`;
      }

      const allSamples = Object.keys(det.sample_word_counts).map(Number).sort((a,b) => a-b);
      const budgets = det.metrics.map(m => m.budget);

      const totalSamples = allSamples.length;
      const totalPages = Math.max(1, Math.ceil(totalSamples / timelinePageSize));
      if (timelinePage > totalPages) timelinePage = totalPages;
      if (timelinePage < 1) timelinePage = 1;

      const startIdx = (timelinePage - 1) * timelinePageSize;
      const endIdx = Math.min(startIdx + timelinePageSize, totalSamples);
      const visibleSamples = allSamples.slice(startIdx, endIdx);

      // Determine label interval for budget header
      let labelInterval = 1;
      if (budgets.length > 50) labelInterval = Math.ceil(budgets.length / 20);
      else if (budgets.length > 25) labelInterval = 2;

      // 1. Render Sticky Header Row with Visible Budget Percentages directly
      const headerRow = document.createElement('div');
      headerRow.className = 'timeline-header-row';

      const corner = document.createElement('div');
      corner.className = 'timeline-corner-cell';
      corner.innerHTML = `<span>Sample</span><span style="color:var(--text-dim)">&rarr; Budget</span>`;
      headerRow.appendChild(corner);

      const headerCells = document.createElement('div');
      headerCells.className = 'timeline-header-cells';

      budgets.forEach((b, i) => {
        const hCell = document.createElement('div');
        const pct = Math.round(b * 100);
        const isLabeled = (budgets.length <= 25) || (i % labelInterval === 0) || (i === budgets.length - 1);

        hCell.className = 'timeline-header-cell' + (isLabeled ? '' : ' is-tick');
        hCell.textContent = isLabeled ? `${pct}%` : '·';
        hCell.title = `Perturbation Budget: ${pct}% (${b.toFixed(3)})`;
        headerCells.appendChild(hCell);
      });

      headerRow.appendChild(headerCells);
      container.appendChild(headerRow);

      // 2. Render Sample Rows
      visibleSamples.forEach(sIdx => {
        const row = document.createElement('div');
        row.className = 'timeline-row';

        const lbl = document.createElement('div');
        lbl.className = 'timeline-sample-lbl';
        lbl.innerHTML = `<span>Sample #${sIdx}</span><span style="font-size:10px; color:var(--text-dim);">${det.sample_word_counts[sIdx]}w</span>`;
        lbl.title = `Click to filter by Sample #${sIdx}`;
        lbl.onclick = () => {
          document.getElementById('filterSample').value = sIdx;
          applyFilters();
          document.getElementById('samplesList').scrollIntoView({ behavior: 'smooth' });
        };
        row.appendChild(lbl);

        const cells = document.createElement('div');
        cells.className = 'timeline-cells';

        budgets.forEach(b => {
          const cell = document.createElement('div');
          cell.className = 'timeline-cell';

          const match = det.raw_rows.find(r => r.s === sIdx && Math.abs(r.b - b) < 0.0001);
          const pct = Math.round(b * 100);

          if (!match) {
            cell.className += ' cell-failed';
            cell.title = `Sample #${sIdx} @ Budget ${pct}%: Not evaluated`;
          } else if (match.ev === 1) {
            cell.className += ' cell-evaded';
            cell.title = `Sample #${sIdx} @ Budget ${pct}%: EVADED (Attack Succeeded)\nInitial Conf: ${(match.oc*100).toFixed(1)}% -> Perturbed: ${(match.pc*100).toFixed(1)}%\nEdits: ${match.rc} words`;
          } else if (!match.ok) {
            cell.className += ' cell-failed';
            cell.title = `Sample #${sIdx} @ Budget ${pct}%: Transformation Failed (0 edits possible)`;
          } else {
            cell.className += ' cell-caught';
            cell.title = `Sample #${sIdx} @ Budget ${pct}%: CAUGHT (Defense Resisted)\nInitial Conf: ${(match.oc*100).toFixed(1)}% -> Perturbed: ${(match.pc*100).toFixed(1)}%\nEdits: ${match.rc} words`;
          }

          cell.onclick = () => {
            selectSlice(det.detector_id, det.attacker_id, false);
            document.getElementById('filterSample').value = sIdx;
            document.getElementById('filterBudget').value = b;
            applyFilters();
            document.getElementById('samplesList').scrollIntoView({ behavior: 'smooth' });
          };

          cells.appendChild(cell);
        });

        row.appendChild(cells);
        container.appendChild(row);
      });

      // 3. Update Timeline Footer Controls (Pagination and stats)
      const stats = document.getElementById('timelineStats');
      const pag = document.getElementById('timelinePagination');
      if (stats && pag && budgets.length > 0) {
        stats.innerHTML = `Showing <strong>${visibleSamples.length}</strong> of <strong>${totalSamples}</strong> samples across <strong>${budgets.length}</strong> budget steps (${(budgets[0]*100).toFixed(0)}% &rarr; ${(budgets[budgets.length-1]*100).toFixed(0)}%) for <strong>${escapeHtml(det.detector_label)} — ${escapeHtml(det.attacker_label)}</strong>.`;

        pag.innerHTML = '';
        if (totalSamples > 25) {
          pag.innerHTML = `
            <span style="font-size:11px; color:var(--text-muted);">Rows:</span>
            <select class="filter-select" style="padding:2px 6px; font-size:11px;" onchange="setTimelinePageSize(this.value)">
              <option value="25" ${timelinePageSize === 25 ? 'selected' : ''}>25</option>
              <option value="50" ${timelinePageSize === 50 ? 'selected' : ''}>50</option>
              <option value="100" ${timelinePageSize === 100 ? 'selected' : ''}>100</option>
              <option value="all" ${timelinePageSize >= 999999 ? 'selected' : ''}>All</option>
            </select>
            <button class="btn" style="padding:2px 8px; font-size:11px;" onclick="changeTimelinePage(-1)" ${timelinePage === 1 ? 'disabled' : ''}>&larr;</button>
            <span style="font-size:11px;">Page ${timelinePage} / ${totalPages}</span>
            <button class="btn" style="padding:2px 8px; font-size:11px;" onclick="changeTimelinePage(1)" ${timelinePage === totalPages ? 'disabled' : ''}>&rarr;</button>
          `;
        }
      }
    }

    // -------------------------------------------------------------------------
    // Filter Dropdowns & Sample Cards
    // -------------------------------------------------------------------------
    function initFilterDropdowns() {
      const bSelect = document.getElementById('filterBudget');
      const sSelect = document.getElementById('filterSample');
      bSelect.innerHTML = '<option value="all">Budget: All Steps</option>';
      sSelect.innerHTML = '<option value="all">Sample: All Samples</option>';

      const sampleDet = BENCHMARK_DATA[0];
      if (!sampleDet) return;
      sampleDet.metrics.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.budget;
        opt.textContent = `Budget: ${(m.budget * 100).toFixed(0)}% (${m.budget.toFixed(2)})`;
        bSelect.appendChild(opt);
      });

      Object.keys(sampleDet.sample_word_counts).map(Number).sort((a,b) => a-b).forEach(idx => {
        const opt = document.createElement('option');
        opt.value = idx;
        opt.textContent = `Sample #${idx} (${sampleDet.sample_word_counts[idx]} words)`;
        sSelect.appendChild(opt);
      });
    }

    function resetFilters() {
      document.getElementById('searchInput').value = '';
      document.getElementById('filterOutcome').value = 'all';
      document.getElementById('filterBudget').value = 'all';
      document.getElementById('filterSample').value = 'all';
      document.getElementById('filterMinDrop').value = '0';
      document.getElementById('sortSelect').value = 'evaded_first';
      applyFilters();
    }

    function handleSearchInput() {
      applyFilters();
    }

    function applyFilters() {
      const q = document.getElementById('searchInput').value.trim().toLowerCase();
      const outcome = document.getElementById('filterOutcome').value;
      const bFilter = document.getElementById('filterBudget').value;
      const sFilter = document.getElementById('filterSample').value;
      const minDrop = parseFloat(document.getElementById('filterMinDrop').value) || 0;
      const sortBy = document.getElementById('sortSelect').value;

      const activeRuns = getActiveRuns();
      let sourceRows = [];
      activeRuns.forEach(d => {
        d.raw_rows.forEach(r => {
          sourceRows.push({
            ...r,
            detectorName: d.detector_label || d.name,
            attackerName: d.attacker_label || '',
            attackCategory: d.attack_category || 'content',
            detectorKey: d.key,
            detectorId: d.detector_id,
            attackerId: d.attacker_id
          });
        });
      });

      filteredRows = sourceRows.filter(r => {
        if (bFilter !== 'all' && Math.abs(r.b - parseFloat(bFilter)) > 0.0001) return false;
        if (sFilter !== 'all' && r.s !== parseInt(sFilter, 10)) return false;
        if (r.cd < minDrop) return false;

        if (outcome === 'evaded' && r.ev !== 1) return false;
        if (outcome === 'caught' && (r.ev === 1 || !r.ok)) return false;
        if (outcome === 'failed' && r.ok) return false;

        if (q) {
          const metaMatch = r.detectorName.toLowerCase().includes(q) || r.attackerName.toLowerCase().includes(q) || r.attackCategory.toLowerCase().includes(q);
          if (!metaMatch) {
            const subMatch = r.sb.some(s => s.orig.toLowerCase().includes(q) || s.pert.toLowerCase().includes(q));
            if (!subMatch) {
              const chunkMatch = r.ch.some(c => c[1].toLowerCase().includes(q) || (c[2] && c[2].toLowerCase().includes(q)));
              if (!chunkMatch) return false;
            }
          }
        }

        return true;
      });

      if (sortBy === 'drop_desc') {
        filteredRows.sort((a,b) => b.cd - a.cd);
      } else if (sortBy === 'conf_asc') {
        filteredRows.sort((a,b) => a.pc - b.pc);
      } else if (sortBy === 'edits_desc') {
        filteredRows.sort((a,b) => b.rc - a.rc);
      } else if (sortBy === 'evaded_first') {
        filteredRows.sort((a,b) => b.ev - a.ev || b.cd - a.cd || a.s - b.s || a.b - b.b);
      } else {
        filteredRows.sort((a,b) => a.s - b.s || a.b - b.b);
      }

      const evadedCount = filteredRows.filter(r => r.ev === 1).length;
      document.getElementById('lblMatchesCount').textContent = filteredRows.length;
      document.getElementById('lblTotalCount').textContent = sourceRows.length;
      document.getElementById('lblEvadedCount').textContent = evadedCount;
      document.getElementById('lblResistedCount').textContent = filteredRows.length - evadedCount;

      currentPage = 1;
      renderSampleCards();
      renderPagination();
    }

    function renderSampleCards() {
      const list = document.getElementById('samplesList');
      list.innerHTML = '';

      if (filteredRows.length === 0) {
        list.innerHTML = `
          <div style="text-align: center; padding: 48px 20px; background: var(--bg-card); border-radius: var(--radius-md); border: 1px dashed var(--border-subtle); color: var(--text-muted);">
            <h3>No iterations match your filter criteria</h3>
            <p style="font-size: 13px; margin-top: 6px;">Try clearing search terms or selecting "All Results".</p>
            <button class="btn btn-primary" style="margin-top: 14px;" onclick="resetFilters()">Reset All Filters</button>
          </div>
        `;
        return;
      }

      const startIdx = (currentPage - 1) * pageSize;
      const endIdx = Math.min(startIdx + pageSize, filteredRows.length);
      const pageRows = filteredRows.slice(startIdx, endIdx);

      pageRows.forEach((r, idx) => {
        const card = document.createElement('article');
        card.className = 'sample-card ' + (r.ev === 1 ? 'is-evaded' : 'is-caught');

        const budgetPct = Math.round(r.b * 100);
        const dropPct = (r.cd * 100).toFixed(1);
        const isEvaded = r.ev === 1;

        let diffHtml = '';
        if (globalDiffMode === 'unified') {
          diffHtml = renderUnifiedDiffHtml(r.ch);
        } else if (globalDiffMode === 'split') {
          diffHtml = renderSplitDiffHtml(r.ch);
        } else if (globalDiffMode === 'subs') {
          diffHtml = renderSubsTableHtml(r.sb);
        } else if (globalDiffMode === 'patch') {
          diffHtml = `<pre class="patch-block">${escapeHtml(r.pa)}</pre>`;
        }

        const actualIdx = startIdx + idx;

        card.innerHTML = `
          <div class="sample-header">
            <div class="sample-title-area">
              <span class="sample-id-badge">Sample #${r.s}</span>
              <span class="budget-badge">Budget: ${budgetPct}%</span>
              <span class="meta-tag" style="padding: 2px 8px; cursor:pointer;" onclick="selectSlice('${r.detectorId}', 'all', false)" title="Filter by Detector">Detector: <strong>${escapeHtml(r.detectorName)}</strong></span>
              <span class="meta-tag" style="padding: 2px 8px; cursor:pointer;" onclick="selectSlice('all', '${r.attackerId}', false)" title="Filter by Attack">Attack: <strong>${escapeHtml(r.attackerName)}</strong></span>
              ${catBadgeHtml(r.attackCategory)}
              <span class="status-badge ${isEvaded ? 'status-evaded' : 'status-caught'}">
                ${isEvaded ? '&#9888; EVADED (Label Flipped)' : '&#10003; CAUGHT (Defense Resisted)'}
              </span>
              ${!r.ok ? '<span class="status-badge" style="background:rgba(100,116,139,0.2); color:#94a3b8;">Transform Failed (0 edits)</span>' : ''}
            </div>

            <div class="sample-metrics-strip">
              <div class="confidence-pill">
                <span style="color:var(--text-muted)">Certainty:</span>
                <strong>${(r.oc * 100).toFixed(1)}%</strong> &rarr; 
                <strong style="color: ${isEvaded ? 'var(--evaded-red)' : 'var(--caught-green)'}">${(r.pc * 100).toFixed(1)}%</strong>
                <span style="color: var(--evaded-red); font-size:11px;">(-${dropPct}%)</span>
              </div>

              <div class="meta-tag">
                <span>Edits: <strong>${r.rc} words</strong></span>
              </div>
            </div>
          </div>

          <div class="sample-body">
            ${diffHtml}
          </div>

          <div class="sample-footer">
            <div style="display:flex; align-items:center; gap:12px;">
              <span>Substitutions: <strong>${r.sb.length}</strong> pairs</span>
              <span>Changes: <strong>${r.wc}</strong> words</span>
            </div>

            <div class="sample-action-btns">
              <button class="btn" style="padding:4px 8px; font-size:11px;" onclick="copySampleText(${actualIdx}, 'orig')">Copy Original</button>
              <button class="btn" style="padding:4px 8px; font-size:11px;" onclick="copySampleText(${actualIdx}, 'pert')">Copy Perturbed</button>
              <button class="btn" style="padding:4px 8px; font-size:11px;" onclick="copySampleText(${actualIdx}, 'patch')">Copy Patch</button>
              <button class="btn btn-primary" style="padding:4px 8px; font-size:11px;" onclick="openRawModal(${actualIdx})">Inspect Raw Data</button>
            </div>
          </div>
        `;

        list.appendChild(card);
      });
    }

    // Diff Renderers
    function renderUnifiedDiffHtml(chunks) {
      let html = '<div class="diff-container">';
      chunks.forEach(c => {
        const tag = c[0];
        if (tag === '=') {
          html += escapeHtml(c[1]);
        } else if (tag === '~') {
          html += `<del class="diff-del">${escapeHtml(c[1])}</del><ins class="diff-ins">${escapeHtml(c[2])}</ins>`;
        } else if (tag === '-') {
          html += `<del class="diff-del">${escapeHtml(c[1])}</del>`;
        } else if (tag === '+') {
          html += `<ins class="diff-ins">${escapeHtml(c[1])}</ins>`;
        }
      });
      html += '</div>';
      return html;
    }

    function renderSplitDiffHtml(chunks) {
      let left = '';
      let right = '';
      chunks.forEach(c => {
        const tag = c[0];
        if (tag === '=') {
          left += escapeHtml(c[1]);
          right += escapeHtml(c[1]);
        } else if (tag === '~') {
          left += `<del class="diff-del">${escapeHtml(c[1])}</del>`;
          right += `<ins class="diff-ins">${escapeHtml(c[2])}</ins>`;
        } else if (tag === '-') {
          left += `<del class="diff-del">${escapeHtml(c[1])}</del>`;
        } else if (tag === '+') {
          right += `<ins class="diff-ins">${escapeHtml(c[1])}</ins>`;
        }
      });

      return `
        <div class="split-diff-grid">
          <div class="split-pane">
            <div class="split-pane-header">
              <span>Original Input (Malicious / Phishing)</span>
            </div>
            ${left}
          </div>
          <div class="split-pane">
            <div class="split-pane-header">
              <span>Perturbed Output (Adversarial Text)</span>
            </div>
            ${right}
          </div>
        </div>
      `;
    }

    function renderSubsTableHtml(subs) {
      if (!subs || subs.length === 0) {
        return `<div class="diff-container" style="color:var(--text-dim); text-align:center;">No word substitutions performed for this budget.</div>`;
      }

      let rows = subs.map((s, idx) => `
        <tr>
          <td style="color:var(--text-dim); width: 40px;">#${idx + 1}</td>
          <td><del class="diff-del">${escapeHtml(s.orig || '<none>')}</del></td>
          <td style="width: 30px; text-align: center; color:var(--text-dim);">&rarr;</td>
          <td><ins class="diff-ins">${escapeHtml(s.pert || '<deleted>')}</ins></td>
        </tr>
      `).join('');

      return `
        <div class="diff-container" style="padding:0; overflow:hidden;">
          <table class="substitutions-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Original Token</th>
                <th></th>
                <th>Adversarial Replacement</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      `;
    }

    function setGlobalDiffMode(mode) {
      globalDiffMode = mode;
      document.querySelectorAll('.diff-mode-btn').forEach(b => b.classList.remove('active'));
      if (mode === 'unified') document.getElementById('btnDiffUnified').classList.add('active');
      if (mode === 'split') document.getElementById('btnDiffSplit').classList.add('active');
      if (mode === 'subs') document.getElementById('btnDiffSubs').classList.add('active');
      if (mode === 'patch') document.getElementById('btnDiffPatch').classList.add('active');
      renderSampleCards();
    }

    function renderPagination() {
      const bar = document.getElementById('paginationBar');
      bar.innerHTML = '';

      const totalPages = Math.ceil(filteredRows.length / pageSize);
      if (totalPages <= 1) return;

      const prevBtn = document.createElement('button');
      prevBtn.className = 'btn';
      prevBtn.textContent = '← Prev';
      prevBtn.disabled = currentPage === 1;
      prevBtn.onclick = () => { currentPage--; renderSampleCards(); renderPagination(); };
      bar.appendChild(prevBtn);

      const span = document.createElement('span');
      span.style.fontSize = '12px';
      span.style.color = 'var(--text-muted)';
      span.textContent = `Page ${currentPage} of ${totalPages}`;
      bar.appendChild(span);

      const nextBtn = document.createElement('button');
      nextBtn.className = 'btn';
      nextBtn.textContent = 'Next →';
      nextBtn.disabled = currentPage === totalPages;
      nextBtn.onclick = () => { currentPage++; renderSampleCards(); renderPagination(); };
      bar.appendChild(nextBtn);
    }

    function copySampleText(rowIdx, type) {
      const r = filteredRows[rowIdx];
      let txt = '';
      if (type === 'orig') {
        txt = r.ch.map(c => (c[0] === '=' || c[0] === '~' || c[0] === '-') ? c[1] : '').join('');
      } else if (type === 'pert') {
        txt = r.ch.map(c => (c[0] === '=' ? c[1] : (c[0] === '~' ? c[2] : (c[0] === '+' ? c[1] : '')))).join('');
      } else if (type === 'patch') {
        txt = r.pa;
      }
      navigator.clipboard.writeText(txt).then(() => showToast(`Copied ${type} text to clipboard!`));
    }

    function openRawModal(rowIdx) {
      const r = filteredRows[rowIdx];
      const modal = document.getElementById('rawModal');
      const title = document.getElementById('modalTitle');
      const code = document.getElementById('modalCode');

      title.textContent = `Sample #${r.s} @ Budget ${Math.round(r.b * 100)}% (${r.detectorName} — ${r.attackerName})`;
      code.textContent = JSON.stringify(r, null, 2);
      modal.classList.add('active');
    }

    function closeModal(e) {
      document.getElementById('rawModal').classList.remove('active');
    }

    function copyModalContent() {
      const txt = document.getElementById('modalCode').textContent;
      navigator.clipboard.writeText(txt).then(() => showToast('Raw JSON copied to clipboard!'));
    }

    function exportMatrixCSV() {
      let headers = ['rank', 'attacker_key', 'attacker_name', 'category'];
      uniqueDetectors.forEach(d => {
        headers.push(`${d.id}_evasion_pct`, `${d.id}_detection_pct`, `${d.id}_conf_drop_pct`);
      });
      headers.push('mean_evasion_pct', 'mean_conf_drop_pct');
      let csv = headers.join(',') + '\\n';

      uniqueAttackers.forEach((att, idx) => {
        let row = [idx + 1, `"${att.id}"`, `"${att.label}"`, `"${att.category}"`];
        uniqueDetectors.forEach(det => {
          const r = runGrid[`${att.id}__${det.id}`];
          if (r) {
            row.push(r.summary.final_evasion_rate, r.summary.final_detection_rate, r.summary.mean_confidence_drop);
          } else {
            row.push('', '', '');
          }
        });
        row.push(att.meanEvasion.toFixed(1), att.meanDrop.toFixed(2));
        csv += row.join(',') + '\\n';
      });

      const blob = new Blob([csv], { type: 'text/csv' });
      downloadBlob(blob, 'adversarial_2d_matrix.csv');
      showToast('Exported 2D Attack × Detector Matrix CSV!');
    }

    function exportFilteredData(format) {
      if (format === 'json') {
        const blob = new Blob([JSON.stringify(filteredRows, null, 2)], { type: 'application/json' });
        downloadBlob(blob, 'adversarial_report_filtered.json');
      } else if (format === 'csv') {
        let csvContent = "detector,attacker,category,budget,sample_idx,orig_pred,pert_pred,orig_conf,pert_conf,conf_drop,evaded,budget_consumed,success\\n";
        filteredRows.forEach(r => {
          csvContent += `"${r.detectorName}","${r.attackerName}","${r.attackCategory}",${r.b},${r.s},${r.op},${r.pp},${r.oc},${r.pc},${r.cd},${r.ev},${r.rc},${r.ok}\\n`;
        });
        const blob = new Blob([csvContent], { type: 'text/csv' });
        downloadBlob(blob, 'adversarial_report_filtered.csv');
      }
      showToast(`Exported ${filteredRows.length} records as ${format.toUpperCase()}!`);
    }

    function downloadBlob(blob, filename) {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }

    function escapeHtml(str) {
      if (str === null || str === undefined) return '';
      return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    // -------------------------------------------------------------------------
    // Static Chart Toggle
    // -------------------------------------------------------------------------
    function setChartMode(mode) {
      chartViewMode = mode;
      document.getElementById('btnChartDynamic').classList.toggle('active', mode === 'interactive');
      document.getElementById('btnChartImage').classList.toggle('active', mode === 'image');
      document.getElementById('interactiveChartWrapper').style.display = mode === 'interactive' ? 'block' : 'none';
      document.getElementById('chartLegend').style.display = mode === 'interactive' ? 'flex' : 'none';
      document.getElementById('imageChartWrapper').style.display = mode === 'image' ? 'block' : 'none';
      if (mode === 'interactive') renderChart();
    }

    function updateStaticChartImage() {
      const img = document.getElementById('staticChartImg');
      const det = getTimelineTargetRun();
      if (det && det.curve_b64) {
        img.src = det.curve_b64;
        document.getElementById('btnChartImage').style.display = 'inline-block';
      } else {
        document.getElementById('btnChartImage').style.display = 'none';
        setChartMode('interactive');
      }
    }

    // -------------------------------------------------------------------------
    // Pure Canvas Interactive Chart Engine (Multi-Mode + Interactive Legend)
    // -------------------------------------------------------------------------
    function setupCanvasResizing() {
      window.addEventListener('resize', () => {
        if (chartViewMode === 'interactive') renderChart();
      });

      const canvas = document.getElementById('chartCanvas');
      canvas.addEventListener('mousemove', handleChartHover);
      canvas.addEventListener('mouseleave', () => {
        document.getElementById('chartTooltip').style.display = 'none';
      });
    }

    let chartDataPoints = [];

    function averageMetricsAcrossRuns(runs) {
      if (!runs || runs.length === 0) return [];
      const refMetrics = runs[0].metrics || [];
      return refMetrics.map((m, idx) => {
        const stepMetrics = runs.map(r => (r.metrics && r.metrics[idx]) ? r.metrics[idx] : m);
        const detRate = stepMetrics.reduce((s, x) => s + x.detection_rate, 0) / stepMetrics.length;
        const conf = stepMetrics.reduce((s, x) => s + x.mean_pert_confidence, 0) / stepMetrics.length;
        const drop = stepMetrics.reduce((s, x) => s + x.mean_confidence_drop, 0) / stepMetrics.length;
        return {
          budget: m.budget,
          detection_rate: detRate,
          mean_pert_confidence: conf,
          mean_confidence_drop: drop,
          se_detection_rate: 0,
          std_pert_confidence: 0
        };
      });
    }

    function buildChartSeries() {
      const seriesModeSel = document.getElementById('chartSeriesMode')?.value || 'auto';
      const activeRuns = getActiveRuns();
      if (activeRuns.length === 0) return { isSingleRun: false, series: [] };

      // Single (Detector, Attack) run selected
      if (activeRuns.length === 1 && seriesModeSel === 'auto') {
        return { isSingleRun: true, singleRun: activeRuns[0], series: [] };
      }

      let mode = seriesModeSel;
      if (mode === 'auto') {
        if (selectedDetectorId === 'all' && selectedAttackerId === 'all') {
          mode = 'top_attacks';
        } else if (selectedDetectorId === 'all' && selectedAttackerId !== 'all') {
          mode = 'by_detector';
        } else if (selectedDetectorId !== 'all' && selectedAttackerId === 'all') {
          mode = 'top_attacks';
        } else {
          mode = 'all_runs';
        }
      }

      if (mode === 'by_detector') {
        const series = uniqueDetectors.map((det, idx) => {
          const runs = activeRuns.filter(r => r.detector_id === det.id);
          return {
            label: det.label,
            color: SERIES_PALETTE[idx % SERIES_PALETTE.length],
            metrics: averageMetricsAcrossRuns(runs),
            onClick: () => selectSlice(det.id, selectedAttackerId, false)
          };
        }).filter(s => s.metrics.length > 0);
        return { isSingleRun: false, series };
      }

      if (mode === 'by_category') {
        const cats = [
          { id: 'structure', label: 'Structure Attacks', color: '#f43f5e' },
          { id: 'content', label: 'Content Attacks', color: '#6366f1' },
          { id: 'metadata', label: 'Metadata Attacks', color: '#f59e0b' }
        ];
        const series = cats.map(c => {
          const runs = activeRuns.filter(r => r.attack_category === c.id);
          return {
            label: `${c.label} (${runs.length} runs)`,
            color: c.color,
            metrics: averageMetricsAcrossRuns(runs),
            onClick: () => {
              const sel = document.getElementById('matrixCategorySelect');
              if (sel) { sel.value = c.id; renderMatrix(); }
            }
          };
        }).filter(s => s.metrics.length > 0);
        return { isSingleRun: false, series };
      }

      if (mode === 'top_attacks' || mode === 'active_attacks') {
        const limit = mode === 'top_attacks' ? 6 : 99;
        // Group activeRuns by attacker
        const attGroups = uniqueAttackers.map(att => {
          const runs = activeRuns.filter(r => r.attacker_id === att.id);
          if (runs.length === 0) return null;
          const avgEv = runs.reduce((s, r) => s + r.summary.final_evasion_rate, 0) / runs.length;
          const avgDrp = runs.reduce((s, r) => s + r.summary.mean_confidence_drop, 0) / runs.length;
          return { att, runs, avgEv, avgDrp };
        }).filter(Boolean);

        attGroups.sort((a, b) => b.avgEv - a.avgEv || b.avgDrp - a.avgDrp);
        let chosen = mode === 'active_attacks'
          ? attGroups.filter(g => g.avgEv > 0 || g.avgDrp > 0.5)
          : attGroups.slice(0, limit);
        if (chosen.length === 0) chosen = attGroups.slice(0, 5);

        const series = chosen.map((g, idx) => ({
          label: `${g.att.label} (${g.avgEv.toFixed(0)}% ev)`,
          color: SERIES_PALETTE[idx % SERIES_PALETTE.length],
          metrics: averageMetricsAcrossRuns(g.runs),
          onClick: () => selectSlice(selectedDetectorId, g.att.id, false)
        }));
        return { isSingleRun: false, series };
      }

      // 'all_runs'
      const sortedRuns = [...activeRuns].sort((a, b) => b.summary.final_evasion_rate - a.summary.final_evasion_rate || b.summary.mean_confidence_drop - a.summary.mean_confidence_drop);
      const series = sortedRuns.map((r, idx) => ({
        label: `${r.detector_label} — ${r.attacker_label} (${r.summary.final_evasion_rate}% ev)`,
        color: SERIES_PALETTE[idx % SERIES_PALETTE.length],
        metrics: r.metrics,
        onClick: () => selectSlice(r.detector_id, r.attacker_id, false)
      }));
      return { isSingleRun: false, series };
    }

    function renderChart() {
      const canvas = document.getElementById('chartCanvas');
      const legendEl = document.getElementById('chartLegend');
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;

      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;

      const ctx = canvas.getContext('2d');
      ctx.scale(dpr, dpr);

      const w = rect.width;
      const h = rect.height;

      ctx.clearRect(0, 0, w, h);

      const padLeft = 60;
      const padRight = 60;
      const padTop = 30;
      const padBottom = 45;

      const plotW = w - padLeft - padRight;
      const plotH = h - padTop - padBottom;

      const yMetricMode = document.getElementById('chartMetricMode')?.value || 'detection';
      const { isSingleRun, singleRun, series } = buildChartSeries();

      // Draw Grid & Axes
      ctx.strokeStyle = '#1e2c47';
      ctx.lineWidth = 1;
      ctx.font = '11px Inter, sans-serif';
      ctx.fillStyle = '#64748b';

      for (let i = 0; i <= 5; i++) {
        const yVal = i * 20;
        const yPos = padTop + plotH - (i / 5) * plotH;

        ctx.beginPath();
        ctx.moveTo(padLeft, yPos);
        ctx.lineTo(w - padRight, yPos);
        ctx.stroke();

        ctx.textAlign = 'right';
        ctx.fillStyle = isSingleRun ? '#f43f5e' : '#cbd5e1';
        ctx.fillText(`${yVal}%`, padLeft - 10, yPos + 4);

        if (isSingleRun) {
          ctx.textAlign = 'left';
          ctx.fillStyle = '#38bdf8';
          ctx.fillText(`${yVal}%`, w - padRight + 10, yPos + 4);
        }
      }

      const refRun = singleRun || BENCHMARK_DATA[0];
      const maxB = refRun?.config?.max_budget || 0.2;
      const steps = 5;

      for (let i = 0; i <= steps; i++) {
        const bPct = (i / steps) * maxB * 100;
        const xPos = padLeft + (i / steps) * plotW;

        ctx.beginPath();
        ctx.moveTo(xPos, padTop + plotH);
        ctx.lineTo(xPos, padTop + plotH + 5);
        ctx.stroke();

        ctx.textAlign = 'center';
        ctx.fillStyle = '#94a3b8';
        ctx.fillText(`${Math.round(bPct)}%`, xPos, padTop + plotH + 20);
      }

      ctx.fillText('Perturbation Budget (%)', padLeft + plotW / 2, h - 8);

      chartDataPoints = [];
      if (legendEl) legendEl.innerHTML = '';

      if (isSingleRun && singleRun) {
        const cur = singleRun;
        const metrics = cur.metrics;
        if (!metrics || metrics.length === 0) return;

        document.getElementById('chartTitle').textContent = `Degradation Curve: ${cur.detector_label} vs ${cur.attacker_label}`;
        document.getElementById('chartSubtitle').textContent = `Solid Red: Detection Rate (±1 SE) | Dashed Cyan: Mean Malicious Confidence (±1σ)`;

        // 1. Shaded Variance Area for Detection Rate (±1 SE)
        ctx.fillStyle = 'rgba(244, 63, 94, 0.15)';
        ctx.beginPath();
        metrics.forEach((m, i) => {
          const x = padLeft + (m.budget / maxB) * plotW;
          const upperDet = Math.min(1.0, m.detection_rate + m.se_detection_rate);
          const y = padTop + plotH - upperDet * plotH;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        for (let i = metrics.length - 1; i >= 0; i--) {
          const m = metrics[i];
          const x = padLeft + (m.budget / maxB) * plotW;
          const lowerDet = Math.max(0.0, m.detection_rate - m.se_detection_rate);
          const y = padTop + plotH - lowerDet * plotH;
          ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.fill();

        // 2. Shaded Variance Area for Mean Malicious Confidence (±1σ)
        ctx.fillStyle = 'rgba(56, 189, 248, 0.12)';
        ctx.beginPath();
        metrics.forEach((m, i) => {
          const x = padLeft + (m.budget / maxB) * plotW;
          const upperConf = Math.min(1.0, m.mean_pert_confidence + m.std_pert_confidence);
          const y = padTop + plotH - upperConf * plotH;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        for (let i = metrics.length - 1; i >= 0; i--) {
          const m = metrics[i];
          const x = padLeft + (m.budget / maxB) * plotW;
          const lowerConf = Math.max(0.0, m.mean_pert_confidence - m.std_pert_confidence);
          const y = padTop + plotH - lowerConf * plotH;
          ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.fill();

        // 3. Line: Detection Rate (%)
        ctx.strokeStyle = '#f43f5e';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        metrics.forEach((m, i) => {
          const x = padLeft + (m.budget / maxB) * plotW;
          const y = padTop + plotH - m.detection_rate * plotH;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();

        // 4. Line: Mean Confidence (%)
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 2.2;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        metrics.forEach((m, i) => {
          const x = padLeft + (m.budget / maxB) * plotW;
          const y = padTop + plotH - m.mean_pert_confidence * plotH;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.setLineDash([]);

        metrics.forEach(m => {
          const x = padLeft + (m.budget / maxB) * plotW;
          const yDet = padTop + plotH - m.detection_rate * plotH;
          const yConf = padTop + plotH - m.mean_pert_confidence * plotH;

          ctx.fillStyle = '#f43f5e';
          ctx.beginPath();
          ctx.arc(x, yDet, 4, 0, Math.PI * 2);
          ctx.fill();

          ctx.fillStyle = '#38bdf8';
          ctx.beginPath();
          ctx.arc(x, yConf, 3.5, 0, Math.PI * 2);
          ctx.fill();

          chartDataPoints.push({
            x,
            y: yDet,
            budget: m.budget,
            detRate: m.detection_rate,
            conf: m.mean_pert_confidence,
            drop: m.mean_confidence_drop,
            detector: `${cur.detector_label} — ${cur.attacker_label}`
          });
        });

        if (legendEl) {
          legendEl.innerHTML = `
            <div class="legend-item"><span class="legend-swatch" style="background:#f43f5e;"></span> Detection Rate (%) &pm; 1 SE</div>
            <div class="legend-item"><span class="legend-swatch" style="background:#38bdf8;"></span> Mean Malicious Confidence (%) &pm; 1&sigma;</div>
          `;
        }
      } else {
        // Multi-Series Overlay
        const metricLabel = yMetricMode === 'evasion'
          ? 'Evasion Rate (%)'
          : (yMetricMode === 'confidence' ? 'Mean Malicious Confidence (%)' : 'Detection Rate (%)');

        document.getElementById('chartTitle').textContent = `Multi-Curve Comparison — ${metricLabel}`;
        document.getElementById('chartSubtitle').textContent = `Showing ${series.length} curve(s) across perturbation budgets. Click any legend item below to drill into that series.`;

        series.forEach(s => {
          const metrics = s.metrics;
          if (!metrics || metrics.length === 0) return;

          ctx.strokeStyle = s.color;
          ctx.lineWidth = 2.5;
          ctx.beginPath();
          metrics.forEach((m, i) => {
            const x = padLeft + (m.budget / maxB) * plotW;
            const val = yMetricMode === 'evasion'
              ? (1.0 - m.detection_rate)
              : (yMetricMode === 'confidence' ? m.mean_pert_confidence : m.detection_rate);
            const y = padTop + plotH - val * plotH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          });
          ctx.stroke();

          metrics.forEach(m => {
            const x = padLeft + (m.budget / maxB) * plotW;
            const val = yMetricMode === 'evasion'
              ? (1.0 - m.detection_rate)
              : (yMetricMode === 'confidence' ? m.mean_pert_confidence : m.detection_rate);
            const y = padTop + plotH - val * plotH;

            ctx.fillStyle = s.color;
            ctx.beginPath();
            ctx.arc(x, y, 3.8, 0, Math.PI * 2);
            ctx.fill();

            chartDataPoints.push({
              x,
              y,
              budget: m.budget,
              detRate: m.detection_rate,
              conf: m.mean_pert_confidence,
              drop: m.mean_confidence_drop,
              detector: s.label
            });
          });

          if (legendEl) {
            const item = document.createElement('div');
            item.className = 'legend-item';
            item.innerHTML = `<span class="legend-swatch" style="background:${s.color};"></span> <span>${escapeHtml(s.label)}</span>`;
            if (s.onClick) item.onclick = s.onClick;
            legendEl.appendChild(item);
          }
        });
      }
    }

    function handleChartHover(e) {
      const canvas = document.getElementById('chartCanvas');
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      let closest = null;
      let minDist = 25;

      chartDataPoints.forEach(pt => {
        const dist = Math.hypot(pt.x - mouseX, pt.y - mouseY);
        if (dist < minDist) {
          minDist = dist;
          closest = pt;
        }
      });

      const tip = document.getElementById('chartTooltip');
      if (closest) {
        tip.style.display = 'block';
        tip.style.left = `${Math.min(rect.width - 220, Math.max(10, closest.x + 14))}px`;
        tip.style.top = `${Math.max(10, closest.y - 10)}px`;
        tip.innerHTML = `
          <strong>${escapeHtml(closest.detector)}</strong><br>
          Budget: <strong>${(closest.budget * 100).toFixed(0)}%</strong><br>
          Evasion Rate: <strong style="color:var(--evaded-red);">${((1 - closest.detRate) * 100).toFixed(1)}%</strong><br>
          Detection Retained: <strong style="color:var(--caught-green);">${(closest.detRate * 100).toFixed(1)}%</strong><br>
          Malicious Conf: <strong style="color:var(--accent-cyan);">${(closest.conf * 100).toFixed(1)}%</strong><br>
          &Delta; Conf Drop: <strong>-${(closest.drop * 100).toFixed(1)}%</strong>
        `;
      } else {
        tip.style.display = 'none';
      }
    }
  </script>
</body>
</html>
"""


def build_html_report(run_title: str, base_dir: Path, detectors_data: list[dict]) -> str:
    """Builds the complete single-file standalone interactive HTML dashboard."""
    # Common metadata
    first_cfg = detectors_data[0]["config"]
    timestamp = first_cfg.get("timestamp", base_dir.name)
    seed = first_cfg.get("seed", 42)
    max_budget = first_cfg.get("max_budget", 0.2)
    budget_steps = first_cfg.get("budget_steps", 20)

    # Summarize unique detectors and attackers across all loaded runs
    unique_det_labels = []
    unique_att_labels = []
    for d in detectors_data:
        dlbl = d.get("detector_label") or d["name"]
        albl = d.get("attacker_label") or d["config"].get("attacker_display_name", "TextFooler")
        if dlbl not in unique_det_labels:
            unique_det_labels.append(dlbl)
        if albl not in unique_att_labels:
            unique_att_labels.append(albl)

    if len(unique_att_labels) == 1:
        attacker_summary = unique_att_labels[0]
    else:
        attacker_summary = f"{len(unique_att_labels)} Attack Algorithms"

    if len(unique_det_labels) == 1:
        detector_summary = unique_det_labels[0]
    else:
        detector_summary = f"{len(unique_det_labels)} Models ({', '.join(unique_det_labels)})"

    # Serialize detector data cleanly for JS
    js_detectors = []
    for d in detectors_data:
        js_detectors.append({
            "key": d["key"],
            "name": d["name"],
            "detector_id": d.get("detector_id", d["key"]),
            "detector_label": d.get("detector_label", d["name"]),
            "attacker_id": d.get("attacker_id", d["config"].get("attacker_key", "textfooler")),
            "attacker_label": d.get("attacker_label", d["config"].get("attacker_display_name", "TextFooler")),
            "attack_category": d.get("attack_category", "content"),
            "config": d["config"],
            "metrics": d["metrics"],
            "summary": d["summary"],
            "curve_b64": d["curve_b64"],
            "sample_word_counts": d["sample_word_counts"],
            "sample_flip_points": d["sample_flip_points"],
            "raw_rows": [
                {
                    "b": r["budget"],
                    "s": r["sample_idx"],
                    "op": r["orig_pred"],
                    "pp": r["pert_pred"],
                    "oc": r["orig_conf"],
                    "pc": r["pert_conf"],
                    "cd": r["conf_drop"],
                    "ev": r["evaded"],
                    "rc": r["real_budget_consumed"],
                    "ok": r["success"],
                    "wc": r["words_changed"],
                    "ch": r["chunks"],
                    "sb": r["subs"],
                    "pa": r["patch"]
                }
                for r in d["raw_rows"]
            ]
        })

    json_payload = json.dumps(js_detectors)

    # Substitute tokens in non-f-string template
    output_html = HTML_TEMPLATE
    output_html = output_html.replace("__RUN_TITLE__", html.escape(run_title))
    output_html = output_html.replace("__ATTACKER_NAME__", html.escape(attacker_summary))
    output_html = output_html.replace("__DETECTOR_SUMMARY__", html.escape(detector_summary))
    output_html = output_html.replace("__RUN_COUNT__", str(len(detectors_data)))
    output_html = output_html.replace("__TIMESTAMP__", html.escape(str(timestamp)))
    output_html = output_html.replace("__MAX_BUDGET_PCT__", str(int(max_budget * 100)))
    output_html = output_html.replace("__BUDGET_STEPS__", str(budget_steps))
    output_html = output_html.replace("__SEED__", str(seed))
    output_html = output_html.replace("__JSON_PAYLOAD__", json_payload)

    return output_html


# ---------------------------------------------------------------------------
# Main Execution Entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate an interactive, comprehensive HTML report from adversarial benchmark results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "results_dir",
        type=str,
        nargs="?",
        default=None,
        help="Path to results directory (timestamp run folder, detector subfolder, or root results/ directory)."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path for generated HTML report (defaults to <results_dir>/report.html)."
    )
    parser.add_argument(
        "--title", "-t",
        type=str,
        default=None,
        help="Custom title for the HTML report."
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Automatically open the generated HTML report in default web browser."
    )

    args = parser.parse_args()

    # Determine default path if none specified
    if not args.results_dir:
        default_root = Path(__file__).resolve().parent / "results"
        if default_root.exists():
            args.results_dir = str(default_root)
        else:
            parser.error("No results directory provided and analysis/results does not exist.")

    base_dir, detector_dirs = resolve_results_directory(args.results_dir)
    print(f"Resolved Benchmark Base: {base_dir}")
    print(f"Found {len(detector_dirs)} detector result dataset(s): {[d.name for d in detector_dirs]}")

    # Load data for each detector
    detectors_data = []
    for d in detector_dirs:
        detectors_data.append(load_detector_run(d))

    if not detectors_data:
        print("Error: No valid detector data could be loaded.")
        sys.exit(1)

    # Determine report title
    first_det = detectors_data[0]
    run_ts = first_det["config"].get("timestamp", base_dir.name)
    unique_dets = {d["detector_id"] for d in detectors_data}
    unique_atts = {d["attacker_id"] for d in detectors_data}

    if args.title:
        title = args.title
    elif len(unique_dets) > 1 and len(unique_atts) > 1:
        title = f"Adversarial Benchmark Matrix: {len(unique_dets)} Detectors × {len(unique_atts)} Attacks ({run_ts})"
    elif len(detectors_data) > 1:
        attacker_name = first_det["attacker_label"]
        title = f"{attacker_name} Multi-Detector Degradation Benchmark ({run_ts})"
    else:
        attacker_name = first_det["attacker_label"]
        title = f"{attacker_name} vs. {first_det['name']} Degradation Analysis ({run_ts})"

    # Generate 2D Summary Tables (summary_table.txt, table.txt, summary_matrix.csv)
    summary_text, matrix_csv_path = generate_summary_tables(base_dir, detectors_data)
    print("\n" + summary_text)

    # Generate HTML
    print("Generating interactive HTML report ...")
    html_content = build_html_report(title, base_dir, detectors_data)

    # Determine output file path
    if args.output:
        out_path = Path(args.output).resolve()
    else:
        out_path = base_dir / "report.html"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\nReport successfully generated!")
    print(f"HTML Report   : {out_path}")
    print(f"Matrix CSV    : {matrix_csv_path}")
    print(f"Summary Table : {base_dir / 'summary_table.txt'}")
    print(f"File Size     : {out_path.stat().st_size / 1024:.1f} KB")

    if args.open:
        import webbrowser
        print(f"Opening report in default browser ...")
        webbrowser.open(out_path.as_uri())

    return str(out_path)


if __name__ == "__main__":
    main()
