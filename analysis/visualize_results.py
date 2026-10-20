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

    /* Multi-Detector Model Selector Tabs */
    .detector-nav {
      display: flex;
      align-items: center;
      gap: 8px;
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 6px;
      margin-bottom: 24px;
      overflow-x: auto;
    }

    .detector-tab {
      padding: 8px 16px;
      border-radius: var(--radius-sm);
      font-size: 13px;
      font-weight: 600;
      color: var(--text-muted);
      background: transparent;
      border: none;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      white-space: nowrap;
      transition: all 0.15s ease;
    }
    .detector-tab:hover {
      color: #fff;
      background: rgba(255, 255, 255, 0.04);
    }
    .detector-tab.active {
      background: var(--bg-card);
      color: #fff;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
      border: 1px solid var(--border-accent);
    }
    .detector-tab-badge {
      font-size: 11px;
      padding: 2px 6px;
      border-radius: 10px;
      background: rgba(255, 255, 255, 0.08);
    }
    .detector-tab.active .detector-tab-badge {
      background: var(--accent-primary);
      color: #fff;
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
        <span class="meta-tag">Attacker: <strong>__ATTACKER_NAME__</strong></span>
        <span class="meta-tag">Timestamp: <strong>__TIMESTAMP__</strong></span>
        <span class="meta-tag">Max Budget: <strong>__MAX_BUDGET_PCT__% (__BUDGET_STEPS__ steps)</strong></span>
        <span class="meta-tag">Seed: <strong>__SEED__</strong></span>
      </div>

      <div class="header-actions">
        <button class="btn btn-primary" onclick="exportFilteredData('csv')">Export CSV</button>
        <button class="btn" onclick="exportFilteredData('json')">Export JSON</button>
      </div>
    </div>
  </header>

  <div class="container">

    <!-- KPI Summary Grid -->
    <section class="kpi-grid" id="kpiGrid">
      <!-- Injected via JavaScript based on selected detector or aggregate -->
    </section>

    <!-- Multi-Detector Model Selector Tabs -->
    <nav class="detector-nav" id="detectorNav">
      <!-- Tabs injected via JavaScript -->
    </nav>

    <!-- Multi-Detector Comparison Section (Shown if multiple detectors) -->
    <section class="comparison-section" id="comparisonSection">
      <div class="section-header">
        <h2 class="section-title">Detector Robustness Comparison Matrix</h2>
        <span class="meta-tag">Evaluating resistance across __BUDGET_STEPS__ perturbation budget levels</span>
      </div>
      <div class="comparison-table-wrapper">
        <table class="data-table" id="comparisonTable">
          <thead>
            <tr>
              <th>Detector Model</th>
              <th>Clean Acc.</th>
              <th>Degraded Acc.</th>
              <th>Overall Evasion</th>
              <th>Resilient Samples</th>
              <th>Mean Conf. Drop</th>
              <th>Avg Budget Consumed</th>
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
      <div class="section-header">
        <div>
          <h2 class="section-title" id="chartTitle">Degradation Curve Analysis</h2>
          <p class="kpi-subtext" id="chartSubtitle">Detection Rate (%) and Mean Malicious Confidence (%) vs Perturbation Budget (%)</p>
        </div>
        <div class="chart-toolbar">
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

      <div id="imageChartWrapper" style="display: none; text-align: center; margin-top: 14px;">
        <img id="staticChartImg" src="" alt="Degradation Curve" style="max-width: 100%; border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
      </div>
    </section>

    <!-- Sample Trajectory Heatmap Grid -->
    <section class="timeline-card">
      <div class="section-header">
        <div>
          <h2 class="section-title">Sample Evasion Progression Timeline</h2>
          <p class="kpi-subtext">Interactive matrix of sample evasion status across budget steps. Top row displays budget percentages directly without hovering. Click any cell to jump to its diff.</p>
        </div>
        <div class="timeline-toolbar">
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
          <input type="text" id="searchInput" class="search-input" placeholder="Search original or perturbed text, substitutions, keywords..." oninput="handleSearchInput()">
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
          <option value="default">Sort: Sample ID & Budget</option>
          <option value="drop_desc">Sort: Largest Confidence Drop (&darr;)</option>
          <option value="conf_asc">Sort: Lowest Perturbed Confidence (&uarr;)</option>
          <option value="edits_desc">Sort: Most Words Modified (&darr;)</option>
          <option value="evaded_first">Sort: Evaded First</option>
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
          <button class="btn" style="padding: 4px 8px; font-size: 11px;" onclick="resetFilters()">Reset Filters</button>
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
    // State
    let currentDetectorIdx = 0; // Index in BENCHMARK_DATA
    let isMultiView = false;
    let globalDiffMode = 'unified'; // 'unified' | 'split' | 'subs' | 'patch'
    let chartViewMode = 'interactive'; // 'interactive' | 'image'
    let currentPage = 1;
    const pageSize = 15;

    let filteredRows = [];

    // Initialize application on load
    window.addEventListener('DOMContentLoaded', () => {
      initDetectorNav();
      initFilterDropdowns();
      switchDetector(0);
      setupCanvasResizing();
    });

    function showToast(msg) {
      const toast = document.getElementById('toast');
      toast.textContent = msg;
      toast.classList.add('show');
      setTimeout(() => toast.classList.remove('show'), 2200);
    }

    // Model Navigation Tabs
    function initDetectorNav() {
      const nav = document.getElementById('detectorNav');
      nav.innerHTML = '';

      if (BENCHMARK_DATA.length > 1) {
        const allTab = document.createElement('button');
        allTab.className = 'detector-tab' + (isMultiView ? ' active' : '');
        allTab.innerHTML = `All Detectors Comparison <span class="detector-tab-badge">${BENCHMARK_DATA.length} Models</span>`;
        allTab.onclick = () => switchDetector(-1);
        nav.appendChild(allTab);
      }

      BENCHMARK_DATA.forEach((det, idx) => {
        const btn = document.createElement('button');
        btn.className = 'detector-tab' + (!isMultiView && currentDetectorIdx === idx ? ' active' : '');
        btn.innerHTML = `${det.name} <span class="detector-tab-badge">${det.summary.final_detection_rate}% Retained</span>`;
        btn.onclick = () => switchDetector(idx);
        nav.appendChild(btn);
      });
    }

    function switchDetector(idx) {
      if (idx === -1) {
        isMultiView = true;
      } else {
        isMultiView = false;
        currentDetectorIdx = idx;
      }

      // Update Nav active classes
      const tabs = document.querySelectorAll('.detector-tab');
      tabs.forEach((tab, i) => {
        if (isMultiView) {
          tab.classList.toggle('active', i === 0);
        } else {
          tab.classList.toggle('active', i === (BENCHMARK_DATA.length > 1 ? idx + 1 : idx));
        }
      });

      renderKPICards();
      renderComparisonTable();
      renderTimelineGrid();
      updateStaticChartImage();
      renderChart();
      applyFilters();
    }

    // Render KPI Cards
    function renderKPICards() {
      const grid = document.getElementById('kpiGrid');
      grid.innerHTML = '';

      let totalSamples, totalEvaded, overallEvasion, meanDrop, avgEdits, failedTransforms;

      if (!isMultiView) {
        const cur = BENCHMARK_DATA[currentDetectorIdx];
        totalSamples = cur.summary.total_samples;
        totalEvaded = cur.summary.total_evaded_samples;
        overallEvasion = cur.summary.final_evasion_rate;
        meanDrop = cur.summary.mean_confidence_drop;
        avgEdits = cur.summary.avg_budget_consumed;
        failedTransforms = cur.summary.total_failed_transforms;
      } else {
        totalSamples = BENCHMARK_DATA.reduce((acc, d) => acc + d.summary.total_samples, 0);
        totalEvaded = BENCHMARK_DATA.reduce((acc, d) => acc + d.summary.total_evaded_samples, 0);
        const sumEvasion = BENCHMARK_DATA.reduce((acc, d) => acc + d.summary.final_evasion_rate, 0);
        overallEvasion = (sumEvasion / BENCHMARK_DATA.length).toFixed(1);
        const sumDrop = BENCHMARK_DATA.reduce((acc, d) => acc + d.summary.mean_confidence_drop, 0);
        meanDrop = (sumDrop / BENCHMARK_DATA.length).toFixed(1);
        const sumEdits = BENCHMARK_DATA.reduce((acc, d) => acc + d.summary.avg_budget_consumed, 0);
        avgEdits = (sumEdits / BENCHMARK_DATA.length).toFixed(1);
        failedTransforms = BENCHMARK_DATA.reduce((acc, d) => acc + d.summary.total_failed_transforms, 0);
      }

      const cards = [
        {
          label: 'Evasion Rate (Attack Success)',
          value: `${overallEvasion}%`,
          subtext: `${totalEvaded} of ${totalSamples} samples evaded`,
          accent: 'var(--evaded-red)'
        },
        {
          label: 'Detector Resiliency (Pass Rate)',
          value: `${(100 - overallEvasion).toFixed(1)}%`,
          subtext: `${totalSamples - totalEvaded} samples remained detected`,
          accent: 'var(--caught-green)'
        },
        {
          label: 'Mean Confidence Drop',
          value: `-${meanDrop}%`,
          subtext: 'Average malicious certainty erosion',
          accent: 'var(--accent-primary)'
        },
        {
          label: 'Avg Words Consumed',
          value: `${avgEdits}`,
          subtext: 'Adversarial word edits per email',
          accent: 'var(--accent-cyan)'
        },
        {
          label: 'Failed Transformations',
          value: `${failedTransforms}`,
          subtext: 'Budget requested but 0 edits possible',
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

    // Multi-Detector Comparison Table
    function renderComparisonTable() {
      const sec = document.getElementById('comparisonSection');
      if (BENCHMARK_DATA.length <= 1) {
        sec.style.display = 'none';
        return;
      }
      sec.style.display = 'block';

      const tbody = document.getElementById('comparisonTableBody');
      tbody.innerHTML = '';

      BENCHMARK_DATA.forEach((det, i) => {
        const tr = document.createElement('tr');
        const s = det.summary;
        const cleanAcc = '100.0%';
        const degAcc = `${s.final_detection_rate}%`;
        const evRate = `${s.final_evasion_rate}%`;
        const flipStr = s.avg_flip_budget ? `${s.avg_flip_budget}% budget` : 'N/A';

        tr.innerHTML = `
          <td><strong>${det.name}</strong></td>
          <td><span style="color: var(--caught-green); font-weight:600;">${cleanAcc}</span></td>
          <td><span style="color: ${s.final_detection_rate < 50 ? 'var(--evaded-red)' : 'var(--caught-green)'}; font-weight:600;">${degAcc}</span></td>
          <td><span class="status-badge ${s.final_evasion_rate > 50 ? 'status-evaded' : 'status-caught'}">${evRate}</span></td>
          <td>${s.resilient_samples} / ${s.total_samples}</td>
          <td>-${s.mean_confidence_drop}%</td>
          <td>${s.avg_budget_consumed} words</td>
          <td>${flipStr}</td>
        `;
        tr.style.cursor = 'pointer';
        tr.onclick = () => switchDetector(i);
        tbody.appendChild(tr);
      });
    }

    // Timeline Progression Grid State & Renderer
    let timelineDensity = 'normal'; // 'compact' | 'normal' | 'wide'
    let timelinePage = 1;
    let timelinePageSize = 50;

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

      const det = isMultiView ? BENCHMARK_DATA[0] : BENCHMARK_DATA[currentDetectorIdx];
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
      if (stats && pag) {
        stats.innerHTML = `Showing <strong>${visibleSamples.length}</strong> of <strong>${totalSamples}</strong> samples across <strong>${budgets.length}</strong> budget steps (${(budgets[0]*100).toFixed(0)}% &rarr; ${(budgets[budgets.length-1]*100).toFixed(0)}%).`;

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

    // Filter Dropdowns
    function initFilterDropdowns() {
      const bSelect = document.getElementById('filterBudget');
      const sSelect = document.getElementById('filterSample');
      bSelect.innerHTML = '<option value="all">Budget: All Steps</option>';
      sSelect.innerHTML = '<option value="all">Sample: All Samples</option>';

      const sampleDet = BENCHMARK_DATA[0];
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
      document.getElementById('sortSelect').value = 'default';
      applyFilters();
    }

    function handleSearchInput() {
      applyFilters();
    }

    // Apply Filters & Sorting
    function applyFilters() {
      const q = document.getElementById('searchInput').value.trim().toLowerCase();
      const outcome = document.getElementById('filterOutcome').value;
      const bFilter = document.getElementById('filterBudget').value;
      const sFilter = document.getElementById('filterSample').value;
      const minDrop = parseFloat(document.getElementById('filterMinDrop').value) || 0;
      const sortBy = document.getElementById('sortSelect').value;

      let sourceRows = [];
      if (isMultiView) {
        BENCHMARK_DATA.forEach(d => {
          d.raw_rows.forEach(r => {
            sourceRows.push({ ...r, detectorName: d.name, detectorKey: d.key });
          });
        });
      } else {
        const cur = BENCHMARK_DATA[currentDetectorIdx];
        sourceRows = cur.raw_rows.map(r => ({ ...r, detectorName: cur.name, detectorKey: cur.key }));
      }

      filteredRows = sourceRows.filter(r => {
        if (bFilter !== 'all' && Math.abs(r.b - parseFloat(bFilter)) > 0.0001) return false;
        if (sFilter !== 'all' && r.s !== parseInt(sFilter, 10)) return false;
        if (r.cd < minDrop) return false;

        if (outcome === 'evaded' && r.ev !== 1) return false;
        if (outcome === 'caught' && (r.ev === 1 || !r.ok)) return false;
        if (outcome === 'failed' && r.ok) return false;

        if (q) {
          // Search in substitutions
          const subMatch = r.sb.some(s => s.orig.toLowerCase().includes(q) || s.pert.toLowerCase().includes(q));
          if (!subMatch) {
            // Search in diff chunks
            const chunkMatch = r.ch.some(c => c[1].toLowerCase().includes(q) || (c[2] && c[2].toLowerCase().includes(q)));
            if (!chunkMatch) return false;
          }
        }

        return true;
      });

      // Sorting
      if (sortBy === 'drop_desc') {
        filteredRows.sort((a,b) => b.cd - a.cd);
      } else if (sortBy === 'conf_asc') {
        filteredRows.sort((a,b) => a.pc - b.pc);
      } else if (sortBy === 'edits_desc') {
        filteredRows.sort((a,b) => b.rc - a.rc);
      } else if (sortBy === 'evaded_first') {
        filteredRows.sort((a,b) => b.ev - a.ev || a.s - b.s || a.b - b.b);
      } else {
        filteredRows.sort((a,b) => a.s - b.s || a.b - b.b);
      }

      // Update counters
      const evadedCount = filteredRows.filter(r => r.ev === 1).length;
      document.getElementById('lblMatchesCount').textContent = filteredRows.length;
      document.getElementById('lblTotalCount').textContent = sourceRows.length;
      document.getElementById('lblEvadedCount').textContent = evadedCount;
      document.getElementById('lblResistedCount').textContent = filteredRows.length - evadedCount;

      currentPage = 1;
      renderSampleCards();
      renderPagination();
    }

    // Render Sample Cards
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
              <span class="meta-tag" style="padding: 2px 8px;"><strong>${r.detectorName}</strong></span>
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

    // Pagination
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

    // Copying Utilities
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

    // Modal Inspector
    function openRawModal(rowIdx) {
      const r = filteredRows[rowIdx];
      const modal = document.getElementById('rawModal');
      const title = document.getElementById('modalTitle');
      const code = document.getElementById('modalCode');

      title.textContent = `Sample #${r.s} @ Budget ${Math.round(r.b * 100)}% (${r.detectorName})`;
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

    // Export Handler
    function exportFilteredData(format) {
      if (format === 'json') {
        const blob = new Blob([JSON.stringify(filteredRows, null, 2)], { type: 'application/json' });
        downloadBlob(blob, 'adversarial_report_filtered.json');
      } else if (format === 'csv') {
        let csvContent = "detector,budget,sample_idx,orig_pred,pert_pred,orig_conf,pert_conf,conf_drop,evaded,budget_consumed,success\\n";
        filteredRows.forEach(r => {
          csvContent += `"${r.detectorName}",${r.b},${r.s},${r.op},${r.pp},${r.oc},${r.pc},${r.cd},${r.ev},${r.rc},${r.ok}\\n`;
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
      if (!str) return '';
      return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    // Static Chart Toggle
    function setChartMode(mode) {
      chartViewMode = mode;
      document.getElementById('btnChartDynamic').classList.toggle('active', mode === 'interactive');
      document.getElementById('btnChartImage').classList.toggle('active', mode === 'image');
      document.getElementById('interactiveChartWrapper').style.display = mode === 'interactive' ? 'block' : 'none';
      document.getElementById('imageChartWrapper').style.display = mode === 'image' ? 'block' : 'none';
      if (mode === 'interactive') renderChart();
    }

    function updateStaticChartImage() {
      const img = document.getElementById('staticChartImg');
      const det = isMultiView ? BENCHMARK_DATA[0] : BENCHMARK_DATA[currentDetectorIdx];
      if (det.curve_b64) {
        img.src = det.curve_b64;
        document.getElementById('btnChartImage').style.display = 'inline-block';
      } else {
        document.getElementById('btnChartImage').style.display = 'none';
        setChartMode('interactive');
      }
    }

    // -------------------------------------------------------------------------
    // Pure Canvas Interactive Chart Engine
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

    let chartDataPoints = []; // Stored points for tooltip hit testing

    function renderChart() {
      const canvas = document.getElementById('chartCanvas');
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

      // Margins
      const padLeft = 60;
      const padRight = 60;
      const padTop = 30;
      const padBottom = 45;

      const plotW = w - padLeft - padRight;
      const plotH = h - padTop - padBottom;

      // Draw Grid & Axes
      ctx.strokeStyle = '#1e2c47';
      ctx.lineWidth = 1;
      ctx.font = '11px Inter, sans-serif';
      ctx.fillStyle = '#64748b';

      // 5 Horizontal Grid lines (0% to 100%)
      for (let i = 0; i <= 5; i++) {
        const yVal = i * 20;
        const yPos = padTop + plotH - (i / 5) * plotH;

        ctx.beginPath();
        ctx.moveTo(padLeft, yPos);
        ctx.lineTo(w - padRight, yPos);
        ctx.stroke();

        // Left Label: Detection Rate (%)
        ctx.textAlign = 'right';
        ctx.fillStyle = '#f43f5e';
        ctx.fillText(`${yVal}%`, padLeft - 10, yPos + 4);

        // Right Label: Confidence (%)
        ctx.textAlign = 'left';
        ctx.fillStyle = '#38bdf8';
        ctx.fillText(`${yVal}%`, w - padRight + 10, yPos + 4);
      }

      // X Axis (Perturbation Budget %)
      const det = isMultiView ? BENCHMARK_DATA[0] : BENCHMARK_DATA[currentDetectorIdx];
      const maxB = det.config.max_budget || 0.2;
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

      // X Axis Label
      ctx.fillText('Perturbation Budget (%)', padLeft + plotW / 2, h - 8);

      chartDataPoints = [];

      if (!isMultiView) {
        // Single Detector Plot: Detection Rate + Confidence Drop + Confidence ±1σ
        const cur = BENCHMARK_DATA[currentDetectorIdx];
        const metrics = cur.metrics;
        if (!metrics || metrics.length === 0) return;

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
        ctx.setLineDash([]); // Reset line dash

        // 5. Store Hit Points
        metrics.forEach(m => {
          const x = padLeft + (m.budget / maxB) * plotW;
          const yDet = padTop + plotH - m.detection_rate * plotH;
          const yConf = padTop + plotH - m.mean_pert_confidence * plotH;

          // Draw point circles
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
            se: m.se_detection_rate,
            std: m.std_pert_confidence,
            detector: cur.name
          });
        });

      } else {
        // Multi-Detector Overlay
        const palette = ['#f43f5e', '#10b981', '#a855f7', '#f59e0b', '#06b6d4'];

        BENCHMARK_DATA.forEach((det, idx) => {
          const color = palette[idx % palette.length];
          const metrics = det.metrics;

          ctx.strokeStyle = color;
          ctx.lineWidth = 2.4;
          ctx.beginPath();
          metrics.forEach((m, i) => {
            const x = padLeft + (m.budget / maxB) * plotW;
            const y = padTop + plotH - m.detection_rate * plotH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          });
          ctx.stroke();

          metrics.forEach(m => {
            const x = padLeft + (m.budget / maxB) * plotW;
            const y = padTop + plotH - m.detection_rate * plotH;
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(x, y, 3.5, 0, Math.PI * 2);
            ctx.fill();

            chartDataPoints.push({
              x,
              y,
              budget: m.budget,
              detRate: m.detection_rate,
              conf: m.mean_pert_confidence,
              drop: m.mean_confidence_drop,
              se: m.se_detection_rate,
              detector: det.name
            });
          });
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
        tip.style.left = `${closest.x + 14}px`;
        tip.style.top = `${closest.y - 10}px`;
        tip.innerHTML = `
          <strong>${closest.detector}</strong><br>
          Budget: <strong>${(closest.budget * 100).toFixed(0)}%</strong><br>
          Detection Rate: <strong style="color:var(--evaded-red);">${(closest.detRate * 100).toFixed(1)}%</strong><br>
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
    attacker_name = first_cfg.get("attacker_display_name", "TextFooler")
    seed = first_cfg.get("seed", 42)
    max_budget = first_cfg.get("max_budget", 0.2)
    budget_steps = first_cfg.get("budget_steps", 20)

    # Serialize detector data cleanly for JS
    js_detectors = []
    for d in detectors_data:
        js_detectors.append({
            "key": d["key"],
            "name": d["name"],
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
    output_html = output_html.replace("__ATTACKER_NAME__", html.escape(attacker_name))
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
    attacker_name = first_det["config"].get("attacker_display_name", "Adversarial Attack")

    if args.title:
        title = args.title
    elif len(detectors_data) > 1:
        title = f"{attacker_name} Multi-Detector Degradation Benchmark ({run_ts})"
    else:
        title = f"{attacker_name} vs. {first_det['name']} Degradation Analysis ({run_ts})"

    # Generate HTML
    print("\nGenerating interactive HTML report ...")
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
    print(f"File Path : {out_path}")
    print(f"File Size : {out_path.stat().st_size / 1024:.1f} KB")

    if args.open:
        import webbrowser
        print(f"Opening report in default browser ...")
        webbrowser.open(out_path.as_uri())

    return str(out_path)


if __name__ == "__main__":
    main()
