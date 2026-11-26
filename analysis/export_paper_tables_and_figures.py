#!/usr/bin/env python3
"""
Export Paper Tables, Telemetry, Case Study, and Figures
=======================================================
Automates Phase 3 manuscript updates after the 100- and 1,000-sample leak-free
holdout benchmark runs complete:
1. Reads all `degradation_metrics.csv`, `attack_attribution.json`, `raw_results.csv`,
   and optional `transfer_results.json` files.
2. Copies all 42+3 `degradation_curve.png` plots into `notebook/Paper/figures/`.
3. Emits a comprehensive JSON + Markdown summary (`paper_export_summary.json` /
   `paper_export_summary.md`) with exact numbers for:
   - Fragility Map (Table V) + Neutral Padding Control comparison
   - Combined Best-of-K Trajectory Checkpoints (Table VI)
   - Constituent Attack Attribution Breakdown (Table VII)
   - Clean multi-tier Qualitative Case Study candidate (Tables VIII-IX)
   - Cross-Detector Transferability Matrices (Tables X-XI)
   - Exact Figure checkpoint captions (0.5%, 5.0%, 10.0%, 20.0%)
"""

import argparse
import base64
import csv
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DETECTOR_ORDER = ["logistic_regression", "char_cnn", "transformer"]

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

TIER_LABELS = {
    "textfooler": "Content (Tier 1)",
    "homoglyph": "Structure (Tier 2)",
    "zero_width": "Structure (Tier 2)",
    "html_entity": "Structure (Tier 3)",
    "html_tag": "Structure (Tier 4)",
    "dom_wrapper": "Structure (Tier 5)",
    "html_comment": "Structure (Tier 5)",
    "sender_header": "Metadata (Tier 6)",
    "auth_results": "Metadata (Tier 6)",
    "received_routing": "Metadata (Tier 6)",
    "x_header": "Metadata (Tier 6)",
    "subject_metadata": "Metadata (Tier 6)",
    "timestamp": "Metadata (Tier 6)",
    "neutral_padding": "Control (Length Dilution)",
    "combine_all": "Combined (Tiers 1--6)",
}


def unwrap_dir(p: Path) -> Path:
    if not p.exists():
        return p
    if any((d / "degradation_metrics.csv").exists() for d in p.iterdir() if d.is_dir()):
        return p
    subs = sorted(
        [
            d
            for d in p.iterdir()
            if d.is_dir()
            and any((sd / "degradation_metrics.csv").exists() for sd in d.iterdir() if sd.is_dir())
        ],
        key=lambda x: x.name,
    )
    return subs[-1] if subs else p


def load_metrics_by_budget(csv_path: Path) -> dict[float, dict]:
    out = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            b = round(float(row["budget"]), 4)
            out[b] = {k: float(v) for k, v in row.items()}
    return out


def compute_attribution_summary(run_dir: Path, num_valid: int) -> list[dict]:
    attr_path = run_dir / "attack_attribution.json"
    if not attr_path.exists():
        return []
    data = json.loads(attr_path.read_text(encoding="utf-8"))
    # Extract final budget (0.2) records per sample
    final_records = {}
    for key, entries in data.items():
        b_str, s_str = key.split("_", 1)
        if abs(float(b_str) - 0.20) < 1e-4:
            final_records[int(s_str)] = entries

    stats = {}
    total_edits = 0
    total_flips = 0
    for s_idx, entries in final_records.items():
        seen_in_sample = set()
        for e in entries:
            att = e["attack"]
            st = stats.setdefault(
                att,
                {"edits": 0, "samples": 0, "conf_drop_sum": 0.0, "flips": 0},
            )
            st["edits"] += 1
            total_edits += 1
            st["conf_drop_sum"] += float(e.get("conf_drop", 0.0))
            seen_in_sample.add(att)
            if e.get("flipped"):
                st["flips"] += 1
                total_flips += 1
        for att in seen_in_sample:
            stats[att]["samples"] += 1

    rows = []
    for att, st in sorted(stats.items(), key=lambda kv: kv[1]["edits"], reverse=True):
        edits = st["edits"]
        flips = st["flips"]
        rows.append(
            {
                "attack": att,
                "edits": edits,
                "edit_share_pct": round(edits / total_edits * 100.0, 2) if total_edits else 0.0,
                "samples_used": st["samples"],
                "sample_share_pct": round(st["samples"] / num_valid * 100.0, 2) if num_valid else 0.0,
                "mean_conf_drop_per_edit_pct": round(-st["conf_drop_sum"] / edits * 100.0, 2) if edits else 0.0,
                "flips": flips,
                "flip_share_pct": round(flips / total_flips * 100.0, 2) if total_flips else 0.0,
                "total_edits_all": total_edits,
                "total_flips_all": total_flips,
            }
        )
    return rows


def find_case_study_candidate(comb_dir: Path) -> dict | None:
    """Find a clean, concise 3-edit multi-tier evasion sample on Transformer combine_all."""
    attr_path = comb_dir / "attack_attribution.json"
    raw_path = comb_dir / "raw_results.csv"
    if not attr_path.exists() or not raw_path.exists():
        return None

    attr = json.loads(attr_path.read_text(encoding="utf-8"))
    final_rows = {}
    with open(raw_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if abs(float(row["budget"]) - 0.20) < 1e-4:
                final_rows[int(row["sample_idx"])] = row

    candidates = []
    for key, entries in attr.items():
        b_str, s_str = key.split("_", 1)
        if abs(float(b_str) - 0.20) >= 1e-4:
            continue
        s_idx = int(s_str)
        if s_idx not in final_rows or int(final_rows[s_idx]["evaded"]) != 1:
            continue
        if 2 <= len(entries) <= 4 and entries[-1].get("flipped"):
            tiers_used = {TIER_LABELS.get(e["attack"], "").split()[0] for e in entries}
            orig_txt = base64.b64decode(final_rows[s_idx]["original_text_b64"]).decode("utf-8")
            pert_txt = base64.b64decode(final_rows[s_idx]["perturbed_text_b64"]).decode("utf-8")
            if 120 <= len(orig_txt) <= 550:
                candidates.append(
                    (
                        len(tiers_used),
                        -abs(len(entries) - 3),
                        float(final_rows[s_idx]["original_confidence"]),
                        s_idx,
                        entries,
                        orig_txt,
                        pert_txt,
                        final_rows[s_idx],
                    )
                )

    if not candidates:
        return None
    candidates.sort(reverse=True)
    _, _, _, s_idx, entries, orig_txt, pert_txt, r = candidates[0]
    return {
        "sample_idx": s_idx,
        "num_edits": len(entries),
        "original_confidence": float(r["original_confidence"]),
        "perturbed_confidence": float(r["perturbed_confidence"]),
        "entries": entries,
        "original_text": orig_txt,
        "perturbed_text": pert_txt,
    }


def export_all(
    single_dir: Path,
    combined_dir: Path,
    transfer_json: Path | None,
    figures_dir: Path,
    out_dir: Path,
    copy_figures: bool = True,
) -> dict:
    single_dir = unwrap_dir(single_dir)
    combined_dir = unwrap_dir(combined_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if copy_figures:
        figures_dir.mkdir(parents=True, exist_ok=True)

    all_keys = SINGLE_ATTACK_KEYS + ["neutral_padding", "combine_all"]
    summary = {
        "valid_counts": {},
        "fragility_20pct": {},
        "figure_checkpoints": {},
        "combine_all_trajectory": {},
        "combine_all_attribution": {},
        "case_study_candidate": None,
        "transferability": None,
    }

    for att in all_keys:
        base = combined_dir if att == "combine_all" else single_dir
        present = all((base / f"{d}_{att}" / "degradation_metrics.csv").exists() for d in DETECTOR_ORDER)
        if not present:
            continue

        summary["fragility_20pct"][att] = {"tier": TIER_LABELS.get(att, "")}
        summary["figure_checkpoints"][att] = {}

        evs, drops = [], []
        for det in DETECTOR_ORDER:
            rdir = base / f"{det}_{att}"
            cfg = json.loads((rdir / "run_config.json").read_text(encoding="utf-8"))
            summary["valid_counts"][det] = int(cfg["num_samples_valid"])
            m = load_metrics_by_budget(rdir / "degradation_metrics.csv")
            m20 = m[0.2]
            ev20 = round(m20["evasion_rate"] * 100.0, 2)
            drp20 = round(-m20["mean_confidence_drop"] * 100.0, 2)
            evs.append(ev20)
            drops.append(drp20)
            summary["fragility_20pct"][att][det] = {
                "evasion_pct": ev20,
                "detection_pct": round(m20["detection_rate"] * 100.0, 2),
                "conf_change_pct": drp20,
                "mean_edits": m20["mean_real_budget_consumed"],
            }
            summary["figure_checkpoints"][att][det] = {
                str(b): {
                    "det_pct": round(m[b]["detection_rate"] * 100.0, 2),
                    "se_pct": round(m[b]["se_detection_rate"] * 100.0, 2),
                    "ev_pct": round(m[b]["evasion_rate"] * 100.0, 2),
                    "conf_change_pct": round(-m[b]["mean_confidence_drop"] * 100.0, 2),
                    "mean_edits": m[b]["mean_real_budget_consumed"],
                }
                for b in (0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2)
                if b in m
            }

            if copy_figures and (rdir / "degradation_curve.png").exists():
                shutil.copy2(
                    rdir / "degradation_curve.png",
                    figures_dir / f"{det}_{att}.png",
                )

        summary["fragility_20pct"][att]["mean_evasion_pct"] = round(sum(evs) / 3.0, 2)
        summary["fragility_20pct"][att]["mean_conf_change_pct"] = round(sum(drops) / 3.0, 2)

    if "combine_all" in summary["figure_checkpoints"]:
        summary["combine_all_trajectory"] = summary["figure_checkpoints"]["combine_all"]
        for det in DETECTOR_ORDER:
            rdir = combined_dir / f"{det}_combine_all"
            summary["combine_all_attribution"][det] = compute_attribution_summary(
                rdir, summary["valid_counts"][det]
            )
        summary["case_study_candidate"] = find_case_study_candidate(
            combined_dir / "transformer_combine_all"
        )

    if transfer_json and transfer_json.exists():
        summary["transferability"] = json.loads(transfer_json.read_text(encoding="utf-8"))

    out_json = out_dir / "paper_export_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved paper export summary to {out_json}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Export paper tables and copy figures from holdout runs.")
    parser.add_argument("--runs-dir", type=str, default=None, help="Unified benchmark directory.")
    parser.add_argument("--single-runs-dir", type=str, default=None, help="Single-attack runs directory.")
    parser.add_argument("--combined-runs-dir", type=str, default=None, help="Combined runs directory.")
    parser.add_argument("--transfer-json", type=str, default=None, help="Path to transfer_results.json.")
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=str(PROJECT_ROOT / "notebook" / "Paper" / "figures"),
        help="Destination directory for paper figures.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "UsefulResullts" / "paper_export"),
        help="Output directory for exported paper JSON.",
    )
    parser.add_argument("--no-copy-figures", action="store_true", help="Do not overwrite Paper/figures PNGs.")
    args = parser.parse_args()

    s_dir = Path(args.runs_dir) if args.runs_dir else Path(args.single_runs_dir)
    c_dir = Path(args.runs_dir) if args.runs_dir else Path(args.combined_runs_dir)
    t_json = Path(args.transfer_json) if args.transfer_json else None

    export_all(
        single_dir=s_dir,
        combined_dir=c_dir,
        transfer_json=t_json,
        figures_dir=Path(args.figures_dir),
        out_dir=Path(args.output_dir),
        copy_figures=not args.no_copy_figures,
    )


if __name__ == "__main__":
    main()
