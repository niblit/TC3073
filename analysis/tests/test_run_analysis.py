import csv
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_analysis import parse_args, run_single_analysis


def test_parse_args_skip_flipped_flags():
    """Verify CLI argument parsing for --skip-flipped and aliases."""
    with patch("sys.argv", ["run_analysis.py"]):
        args = parse_args()
        assert args.skip_flipped is False

    with patch("sys.argv", ["run_analysis.py", "--skip-flipped"]):
        args = parse_args()
        assert args.skip_flipped is True

    with patch("sys.argv", ["run_analysis.py", "--skip-if-flipped"]):
        args = parse_args()
        assert args.skip_flipped is True

    with patch("sys.argv", ["run_analysis.py", "--early-stop"]):
        args = parse_args()
        assert args.skip_flipped is True

    with patch("sys.argv", ["run_analysis.py", "--no-skip-flipped"]):
        args = parse_args()
        assert args.skip_flipped is False


def test_skip_flipped_skips_already_flipped_samples(tmp_path):
    """
    Verify that when skip_flipped=True, samples that flipped detection
    in an earlier budget step are skipped in subsequent steps.
    """
    data_file = tmp_path / "texts.json"
    data_file.write_text(json.dumps([
        {"text": "Sample zero phishing lure", "label": 1},
        {"text": "Sample one suspicious message", "label": 1},
    ]))

    output_dir = tmp_path / "run_skip_true"

    attack_calls = []

    def mock_attack_text(text, budget_frac, attacker):
        attack_calls.append((text, budget_frac))
        return f"perturbed: {text}", 2, True

    mock_detector = MagicMock()

    # Step-by-step predictions:
    # 1. load_malicious_samples calls predict for sample 0: [1]
    # 2. load_malicious_samples calls predict for sample 1: [1]
    # 3. original_preds: predict([s0, s1]) -> [1, 1]
    # 4. Sweep Step 1: active=[0, 1] -> sample 0 flips (0), sample 1 does not (1) -> [0, 1]
    # 5. Sweep Step 2: active=[1] -> sample 1 flips (0) -> [0]
    # (Sweep Step 3: active=[] -> no predict call)
    predict_responses = [
        [1],     # load sample 0
        [1],     # load sample 1
        [1, 1],  # original_preds
        [0, 1],  # step 1 sweep: sample 0 flips
        [0],     # step 2 sweep: sample 1 flips
    ]

    proba_responses = [
        [[0.1, 0.9], [0.2, 0.8]],  # original_probas
        [[0.8, 0.2], [0.3, 0.7]],  # step 1 sweep probas
        [[0.9, 0.1]],              # step 2 sweep probas
    ]

    def mock_predict(texts):
        if not texts:
            return []
        if predict_responses:
            return predict_responses.pop(0)
        return [0] * len(texts)

    def mock_predict_proba(texts):
        if not texts:
            return []
        if proba_responses:
            return proba_responses.pop(0)
        return [[0.9, 0.1]] * len(texts)

    mock_detector.predict.side_effect = mock_predict
    mock_detector.predict_proba.side_effect = mock_predict_proba

    with patch("run_analysis.load_detector") as mock_load_det, \
         patch("run_analysis.attack_text", side_effect=mock_attack_text):

        mock_load_det.return_value = (
            mock_detector,
            "mock_det",
            {"display_name": "Mock Detector"},
            Path("/mock/model")
        )

        res_dir = run_single_analysis(
            detector_type="logistic_regression",
            attacker_type="sender_header",
            num_samples=2,
            max_budget=0.30,
            budget_steps=3,
            seed=42,
            data_path=data_file,
            output_dir=output_dir,
            skip_flipped=True,
        )

        assert res_dir is not None
        assert res_dir.exists()

        # Step 1: 2 samples attacked
        # Step 2: sample 0 skipped, sample 1 attacked (1 sample)
        # Step 3: both samples skipped (0 samples)
        # Total attack calls = 2 + 1 + 0 = 3
        assert len(attack_calls) == 3

        # Verify raw_results.csv contains all 6 rows (3 steps * 2 samples)
        raw_csv = output_dir / "raw_results.csv"
        assert raw_csv.exists()
        with open(raw_csv, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 6

        # In step 3 (rows 4 and 5), both samples should be recorded as evaded
        assert rows[4]["evaded"] == "1"
        assert rows[4]["perturbed_pred"] == "0"
        assert rows[5]["evaded"] == "1"
        assert rows[5]["perturbed_pred"] == "0"

        # Verify degradation_metrics.csv contains all 3 steps
        metrics_csv = output_dir / "degradation_metrics.csv"
        assert metrics_csv.exists()
        with open(metrics_csv, "r", encoding="utf-8") as f:
            metrics_rows = list(csv.DictReader(f))
        assert len(metrics_rows) == 3

        # Step 1: 1 evasion out of 2 -> detection_rate = 0.5
        assert float(metrics_rows[0]["detection_rate"]) == 0.5
        # Step 2: 2 evasions out of 2 -> detection_rate = 0.0
        assert float(metrics_rows[1]["detection_rate"]) == 0.0
        # Step 3: still 0.0 (all flipped)
        assert float(metrics_rows[2]["detection_rate"]) == 0.0

        # Verify run_config.json
        cfg_file = output_dir / "run_config.json"
        assert cfg_file.exists()
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        assert cfg["skip_flipped"] is True


def test_skip_flipped_false_attacks_every_sample(tmp_path):
    """
    Verify that when skip_flipped=False, all samples are attacked on every step,
    even if they already flipped detection in an earlier step.
    """
    data_file = tmp_path / "texts.json"
    data_file.write_text(json.dumps([
        {"text": "Sample zero phishing lure", "label": 1},
        {"text": "Sample one suspicious message", "label": 1},
    ]))

    output_dir = tmp_path / "run_skip_false"
    attack_calls = []

    def mock_attack_text(text, budget_frac, attacker):
        attack_calls.append((text, budget_frac))
        return f"perturbed: {text}", 2, True

    mock_detector = MagicMock()

    predict_responses = [
        [1],     # load sample 0
        [1],     # load sample 1
        [1, 1],  # original_preds
        [0, 0],  # step 1 sweep: both flip
        [0, 0],  # step 2 sweep: both attacked again
        [0, 0],  # step 3 sweep: both attacked again
    ]

    proba_responses = [
        [[0.1, 0.9], [0.2, 0.8]],  # original_probas
        [[0.8, 0.2], [0.8, 0.2]],  # step 1
        [[0.8, 0.2], [0.8, 0.2]],  # step 2
        [[0.8, 0.2], [0.8, 0.2]],  # step 3
    ]

    mock_detector.predict.side_effect = lambda texts: predict_responses.pop(0) if predict_responses else [0] * len(texts)
    mock_detector.predict_proba.side_effect = lambda texts: proba_responses.pop(0) if proba_responses else [[0.9, 0.1]] * len(texts)

    with patch("run_analysis.load_detector") as mock_load_det, \
         patch("run_analysis.attack_text", side_effect=mock_attack_text):

        mock_load_det.return_value = (
            mock_detector,
            "mock_det",
            {"display_name": "Mock Detector"},
            Path("/mock/model")
        )

        res_dir = run_single_analysis(
            detector_type="logistic_regression",
            attacker_type="sender_header",
            num_samples=2,
            max_budget=0.30,
            budget_steps=3,
            seed=42,
            data_path=data_file,
            output_dir=output_dir,
            skip_flipped=False,
        )

        assert res_dir is not None
        # 2 samples * 3 steps = 6 calls
        assert len(attack_calls) == 6

        cfg_file = output_dir / "run_config.json"
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        assert cfg["skip_flipped"] is False


def test_skip_flipped_all_flipped_at_step_one(tmp_path):
    """
    Verify that if ALL samples flip at step 1, steps 2 and 3 perform 0 attack calls.
    """
    data_file = tmp_path / "texts.json"
    data_file.write_text(json.dumps([
        {"text": "Sample zero phishing lure", "label": 1},
        {"text": "Sample one suspicious message", "label": 1},
    ]))

    output_dir = tmp_path / "run_skip_all"
    attack_calls = []

    def mock_attack_text(text, budget_frac, attacker):
        attack_calls.append((text, budget_frac))
        return f"perturbed: {text}", 2, True

    mock_detector = MagicMock()

    predict_responses = [
        [1],     # load sample 0
        [1],     # load sample 1
        [1, 1],  # original_preds
        [0, 0],  # step 1 sweep: BOTH flip to 0
    ]

    proba_responses = [
        [[0.1, 0.9], [0.2, 0.8]],  # original_probas
        [[0.8, 0.2], [0.8, 0.2]],  # step 1 sweep probas
    ]

    mock_detector.predict.side_effect = lambda texts: predict_responses.pop(0) if predict_responses else [0] * len(texts)
    mock_detector.predict_proba.side_effect = lambda texts: proba_responses.pop(0) if proba_responses else [[0.9, 0.1]] * len(texts)

    with patch("run_analysis.load_detector") as mock_load_det, \
         patch("run_analysis.attack_text", side_effect=mock_attack_text):

        mock_load_det.return_value = (
            mock_detector,
            "mock_det",
            {"display_name": "Mock Detector"},
            Path("/mock/model")
        )

        res_dir = run_single_analysis(
            detector_type="logistic_regression",
            attacker_type="sender_header",
            num_samples=2,
            max_budget=0.30,
            budget_steps=4,
            seed=42,
            data_path=data_file,
            output_dir=output_dir,
            skip_flipped=True,
        )

        assert res_dir is not None
        # Only step 1 ran attacks (2 calls); steps 2, 3, 4 performed 0 attacks!
        assert len(attack_calls) == 2

        # All 4 steps * 2 samples = 8 rows
        raw_csv = output_dir / "raw_results.csv"
        with open(raw_csv, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 8
        for r in rows:
            assert r["evaded"] == "1"

        metrics_csv = output_dir / "degradation_metrics.csv"
        with open(metrics_csv, "r", encoding="utf-8") as f:
            metrics_rows = list(csv.DictReader(f))
        assert len(metrics_rows) == 4
        for mr in metrics_rows:
            assert float(mr["detection_rate"]) == 0.0
            assert float(mr["evasion_rate"]) == 1.0
