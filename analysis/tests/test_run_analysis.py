import csv
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_analysis import parse_args, run_single_analysis, WorkPool, attack_text


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
    # 1. original_preds: predict([s0, s1]) -> [1, 1]
    # 2. Sweep Step 1: active=[0, 1] -> sample 0 flips (0), sample 1 does not (1) -> [0, 1]
    # 3. Sweep Step 2: active=[1] -> sample 1 flips (0) -> [0]
    # (Sweep Step 3: active=[] -> no predict call)
    predict_responses = [
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


def test_parse_args_parallel_and_word_ranking_flags():
    """Verify CLI argument parsing for --parallel, --workers, and --word-ranking toggles."""
    with patch("sys.argv", ["run_analysis.py"]):
        args = parse_args()
        assert args.parallel is False
        assert args.workers is None
        assert args.word_ranking is False

    with patch("sys.argv", ["run_analysis.py", "--parallel", "--workers", "6"]):
        args = parse_args()
        assert args.parallel is True
        assert args.workers == 6

    with patch("sys.argv", ["run_analysis.py", "--work-pool", "-w", "4"]):
        args = parse_args()
        assert args.parallel is True
        assert args.workers == 4

    with patch("sys.argv", ["run_analysis.py", "--no-parallel"]):
        args = parse_args()
        assert args.parallel is False

    with patch("sys.argv", ["run_analysis.py", "--word-ranking"]):
        args = parse_args()
        assert args.word_ranking is True

    with patch("sys.argv", ["run_analysis.py", "--precompute-word-ranking"]):
        args = parse_args()
        assert args.word_ranking is True

    with patch("sys.argv", ["run_analysis.py", "--shared-word-ranking"]):
        args = parse_args()
        assert args.word_ranking is True

    with patch("sys.argv", ["run_analysis.py", "--no-word-ranking"]):
        args = parse_args()
        assert args.word_ranking is False


def test_work_pool_preserves_order_and_executes_concurrently():
    """Verify WorkPool executes tasks across workers and returns results in exact input order."""
    import time

    def slow_for_even(x: int) -> int:
        if x % 2 == 0:
            time.sleep(0.02)
        return x * 10

    items = list(range(8))
    with WorkPool(max_workers=4) as pool:
        results = pool.map(slow_for_even, items)

    assert results == [x * 10 for x in items]
    assert WorkPool(max_workers=2).map(lambda x: x, []) == []


def test_parallel_work_pool_execution_in_analysis(tmp_path):
    """
    Verify that when parallel=True is enabled in run_single_analysis, tasks are
    drawn from the work pool, results match sequential ordering, and run_config.json
    records parallel=True and num_workers.
    """
    data_file = tmp_path / "texts.json"
    data_file.write_text(json.dumps([
        {"text": "Sample zero phishing lure", "label": 1},
        {"text": "Sample one suspicious message", "label": 1},
        {"text": "Sample two urgent verification", "label": 1},
    ]))

    output_dir = tmp_path / "run_parallel"
    attack_calls = []

    def mock_attack_text(text, budget_frac, attacker, word_ranking=None):
        attack_calls.append((text, budget_frac, word_ranking))
        return f"perturbed: {text}", 1, True

    mock_detector = MagicMock()
    mock_detector.predict.side_effect = lambda texts: [1] * len(texts)
    mock_detector.predict_proba.side_effect = lambda texts: [[0.2, 0.8]] * len(texts)

    with patch("run_analysis.load_detector") as mock_load_det, \
         patch("run_analysis.attack_text", side_effect=mock_attack_text):

        mock_load_det.return_value = (
            mock_detector,
            "mock_det",
            {"display_name": "Mock Detector"},
            Path("/mock/model"),
        )

        res_dir = run_single_analysis(
            detector_type="logistic_regression",
            attacker_type="zero_width",
            num_samples=3,
            max_budget=0.20,
            budget_steps=2,
            seed=42,
            data_path=data_file,
            output_dir=output_dir,
            parallel=True,
            num_workers=3,
        )

        assert res_dir is not None
        assert len(attack_calls) == 6  # 3 samples * 2 steps

        cfg_file = output_dir / "run_config.json"
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        assert cfg["parallel"] is True
        assert cfg["num_workers"] == 3
        assert cfg["word_ranking"] is False

        raw_csv = output_dir / "raw_results.csv"
        with open(raw_csv, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 6
        # Verify sample indices in CSV are ordered 0, 1, 2 for each budget step
        assert [int(r["sample_idx"]) for r in rows] == [0, 1, 2, 0, 1, 2]


def test_word_ranking_precomputed_once_and_shared_across_steps_and_attackers(tmp_path):
    """
    Verify that when word_ranking=True:
    1. compute_word_ranking is called once per sample (not repeated on every budget step).
    2. The computed ranking is passed to attack_text / attacker.transform on every step.
    3. When shared_word_rankings dict is passed to a second attacker run on the same detector,
       compute_word_ranking is NOT called again (0 extra ranking computations).
    """
    data_file = tmp_path / "texts.json"
    data_file.write_text(json.dumps([
        {"text": "Please verify account password immediately", "label": 1},
        {"text": "Urgent action required for login credentials", "label": 1},
    ]))

    received_rankings = []

    def mock_attack_text(text, budget_frac, attacker, word_ranking=None):
        received_rankings.append((text, budget_frac, word_ranking))
        return f"perturbed: {text}", 1, True

    fake_ranking_fn = MagicMock(side_effect=lambda text, oracle: [(3, 0.85), (1, 0.45)])

    mock_detector = MagicMock()
    mock_detector.predict.side_effect = lambda texts: [1] * len(texts)
    mock_detector.predict_proba.side_effect = lambda texts: [[0.1, 0.9]] * len(texts)

    shared_rankings = {}

    with patch("run_analysis.load_detector") as mock_load_det, \
         patch("run_analysis.attack_text", side_effect=mock_attack_text), \
         patch("run_analysis.compute_word_ranking", side_effect=fake_ranking_fn):

        mock_load_det.return_value = (
            mock_detector,
            "mock_det",
            {"display_name": "Mock Detector"},
            Path("/mock/model"),
        )

        # Run 1: Zero-Width attack with 3 budget steps
        out_dir_1 = tmp_path / "run_att_1"
        run_single_analysis(
            detector_type="logistic_regression",
            attacker_type="zero_width",
            num_samples=2,
            max_budget=0.30,
            budget_steps=3,
            seed=42,
            data_path=data_file,
            output_dir=out_dir_1,
            parallel=True,
            num_workers=2,
            word_ranking=True,
            word_rankings=shared_rankings,
        )

        # Ranking was computed ONLY once per sample (2 samples total), despite 3 budget steps!
        assert fake_ranking_fn.call_count == 2
        assert len(shared_rankings) == 2

        # All 6 attack calls (2 samples * 3 steps) received the precomputed word_ranking!
        assert len(received_rankings) == 6
        for _text, _b, rank in received_rankings:
            assert rank == [(3, 0.85), (1, 0.45)]

        # Run 2: Homoglyph attack sharing `shared_rankings`
        out_dir_2 = tmp_path / "run_att_2"
        run_single_analysis(
            detector_type="logistic_regression",
            attacker_type="homoglyph",
            num_samples=2,
            max_budget=0.30,
            budget_steps=3,
            seed=42,
            data_path=data_file,
            output_dir=out_dir_2,
            parallel=True,
            num_workers=2,
            word_ranking=True,
            word_rankings=shared_rankings,
        )

        # Zero additional calls to compute_word_ranking on the 2nd attacker!
        assert fake_ranking_fn.call_count == 2
        assert len(received_rankings) == 12

        cfg_file = out_dir_2 / "run_config.json"
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        assert cfg["word_ranking"] is True
        assert cfg["parallel"] is True


def test_visualize_results_2d_matrix_and_html_report(tmp_path):
    """
    Verify that load_detector_run, generate_summary_tables, and build_html_report
    accurately parse multi-detector x multi-attacker runs, rank attacks by mean evasion,
    produce summary_matrix.csv + summary_table.txt, and embed 2D matrix metadata in HTML.
    """
    from visualize_results import (
        _infer_detector_and_attacker,
        load_detector_run,
        generate_summary_tables,
        build_html_report,
    )

    # 1. Test _infer_detector_and_attacker on legacy folder names (without base_detector_key)
    inferred = _infer_detector_and_attacker(
        {
            "detector_key": "char_cnn_html_tag",
            "detector_display_name": "CharCNN - HTML Tag Injection Attack",
            "attacker_key": "html_tag",
            "attacker_display_name": "HTML Tag Injection Attack",
        },
        "char_cnn_html_tag",
    )
    assert inferred["detector_id"] == "char_cnn"
    assert inferred["detector_label"] == "CharCNN"
    assert inferred["attacker_id"] == "html_tag"
    assert inferred["attacker_label"] == "HTML Tag Injection Attack"
    assert inferred["attack_category"] == "structure"

    # 2. Create synthetic 2 detectors x 2 attacks benchmark directory
    combos = [
        ("logistic_regression", "Logistic Regression (TF-IDF)", "html_tag", "HTML Tag Injection Attack", 0.2, 0.45),
        ("transformer", "Transformer (DistilBERT)", "html_tag", "HTML Tag Injection Attack", 0.0, 0.65),
        ("logistic_regression", "Logistic Regression (TF-IDF)", "sender_header", "Sender Header Attack", 1.0, 0.03),
        ("transformer", "Transformer (DistilBERT)", "sender_header", "Sender Header Attack", 0.9, 0.05),
    ]

    loaded_runs = []
    for d_key, d_name, a_key, a_name, det_rate, conf_drop in combos:
        folder = tmp_path / f"{d_key}_{a_key}"
        folder.mkdir(parents=True, exist_ok=True)

        (folder / "run_config.json").write_text(json.dumps({
            "timestamp": "20261111-160144",
            "detector_key": f"{d_key}_{a_key}",
            "detector_display_name": f"{d_name} - {a_name}",
            "attacker_key": a_key,
            "attacker_display_name": a_name,
            "num_samples": 2,
            "max_budget": 0.2,
            "budget_steps": 1,
            "seed": 42,
        }), encoding="utf-8")

        with open(folder / "degradation_metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "budget", "num_samples", "detection_rate", "evasion_rate", "se_detection_rate",
                "mean_orig_confidence", "mean_pert_confidence", "std_pert_confidence",
                "mean_confidence_drop", "std_confidence_drop", "mean_real_budget_consumed", "attack_success_rate"
            ])
            writer.writeheader()
            writer.writerow({
                "budget": 0.2,
                "num_samples": 2,
                "detection_rate": det_rate,
                "evasion_rate": round(1.0 - det_rate, 4),
                "se_detection_rate": 0.0,
                "mean_orig_confidence": 0.9,
                "mean_pert_confidence": round(0.9 - conf_drop, 4),
                "std_pert_confidence": 0.05,
                "mean_confidence_drop": conf_drop,
                "std_confidence_drop": 0.02,
                "mean_real_budget_consumed": 3.0,
                "attack_success_rate": 1.0,
            })

        with open(folder / "raw_results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "budget", "sample_idx", "original_text", "perturbed_text",
                "original_pred", "perturbed_pred", "original_confidence", "perturbed_confidence",
                "confidence_drop", "evaded", "real_budget_consumed", "success"
            ])
            writer.writeheader()
            for s_idx in range(2):
                ev = 1 if (det_rate == 0.0 or (det_rate < 1.0 and s_idx == 0)) else 0
                writer.writerow({
                    "budget": 0.2,
                    "sample_idx": s_idx,
                    "original_text": "Urgent verify password",
                    "perturbed_text": "Urgent <span>verify</span> password",
                    "original_pred": 1,
                    "perturbed_pred": 0 if ev else 1,
                    "original_confidence": 0.9,
                    "perturbed_confidence": round(0.9 - conf_drop, 4),
                    "confidence_drop": conf_drop,
                    "evaded": ev,
                    "real_budget_consumed": 3,
                    "success": True,
                })

        loaded_runs.append(load_detector_run(folder))

    # 3. Verify generate_summary_tables creates ranked 2D matrix CSV & text summary
    summary_txt, matrix_csv = generate_summary_tables(tmp_path, loaded_runs)
    assert matrix_csv.exists()
    assert (tmp_path / "summary_table.txt").exists()
    assert (tmp_path / "table.txt").exists()
    assert "HTML Tag Injection Attack" in summary_txt
    assert "2D ATTACK × DETECTOR MATRIX" in summary_txt

    with open(matrix_csv, "r", encoding="utf-8") as f:
        matrix_rows = list(csv.DictReader(f))
    assert len(matrix_rows) == 2
    # Rank 1 should be html_tag (90% mean evasion) over sender_header (5% mean evasion)
    assert matrix_rows[0]["attacker_key"] == "html_tag"
    assert float(matrix_rows[0]["mean_evasion_pct"]) == 90.0
    assert matrix_rows[1]["attacker_key"] == "sender_header"
    assert float(matrix_rows[1]["mean_evasion_pct"]) == 5.0

    # 4. Verify build_html_report includes multi-detector and multi-attacker metadata
    html_out = build_html_report("Test Benchmark Matrix", tmp_path, loaded_runs)
    assert "2 Attack Algorithms" in html_out
    assert "2 Models (Logistic Regression (TF-IDF), Transformer (DistilBERT))" in html_out
    assert "2D Attack &times; Detector Effectiveness Matrix" in html_out
    assert '"detector_id": "logistic_regression"' in html_out
    assert '"attacker_id": "html_tag"' in html_out
    assert '"attack_category": "structure"' in html_out


def test_sample_selection_rejects_unclassified_without_refilling_and_preserves_sample_idx(tmp_path):
    """
    Verify that:
    1. Setting num_samples=3 picks 3 random samples as a maximum and rejects any that fail
       initial classification without pulling extra samples from the dataset to reach 3.
    2. 'sample_idx' in raw_results.csv preserves each sample's original index from the initial
       n random samples, so 'Sample #k' refers to the exact same text across different detectors.
    """
    import base64

    data_file = tmp_path / "texts.json"
    dataset = [
        {"text": f"Malicious sample text number {i}", "label": 1}
        for i in range(10)
    ]
    data_file.write_text(json.dumps(dataset))

    # Detector A classifies sample 0 and sample 2 as malicious, but fails on sample 1 -> [1, 0, 1]
    det_a = MagicMock()
    det_a.predict.side_effect = [
        [1, 0, 1],  # original_preds for 3 picked samples
        [1, 1],     # step 1 sweep on the 2 valid samples
    ]
    det_a.predict_proba.side_effect = [
        [[0.1, 0.9], [0.8, 0.2], [0.2, 0.8]],
        [[0.2, 0.8], [0.3, 0.7]],
    ]

    # Detector B fails on sample 0, but classifies sample 1 and sample 2 as malicious -> [0, 1, 1]
    det_b = MagicMock()
    det_b.predict.side_effect = [
        [0, 1, 1],  # original_preds for 3 picked samples
        [1, 1],     # step 1 sweep on the 2 valid samples
    ]
    det_b.predict_proba.side_effect = [
        [[0.7, 0.3], [0.1, 0.9], [0.2, 0.8]],
        [[0.2, 0.8], [0.3, 0.7]],
    ]

    def mock_attack_text(text, budget_frac, attacker):
        return f"perturbed: {text}", 1, True

    out_a = tmp_path / "det_a"
    out_b = tmp_path / "det_b"

    with patch("run_analysis.attack_text", side_effect=mock_attack_text):
        with patch("run_analysis.load_detector", return_value=(det_a, "det_a", {"display_name": "Det A"}, Path("/mock/a"))):
            run_single_analysis(
                detector_type="logistic_regression",
                attacker_type="sender_header",
                num_samples=3,
                max_budget=0.10,
                budget_steps=1,
                seed=42,
                data_path=data_file,
                output_dir=out_a,
            )

        with patch("run_analysis.load_detector", return_value=(det_b, "det_b", {"display_name": "Det B"}, Path("/mock/b"))):
            run_single_analysis(
                detector_type="char_cnn",
                attacker_type="sender_header",
                num_samples=3,
                max_budget=0.10,
                budget_steps=1,
                seed=42,
                data_path=data_file,
                output_dir=out_b,
            )

    cfg_a = json.loads((out_a / "run_config.json").read_text(encoding="utf-8"))
    cfg_b = json.loads((out_b / "run_config.json").read_text(encoding="utf-8"))

    # Both requested 3 samples, and each evaluated on only 2 valid samples (did NOT refill from remaining 7 dataset items)
    assert cfg_a["num_samples_requested"] == 3
    assert cfg_a["num_samples_valid"] == 2
    assert cfg_b["num_samples_requested"] == 3
    assert cfg_b["num_samples_valid"] == 2

    with open(out_a / "raw_results.csv", "r", encoding="utf-8") as f:
        rows_a = {int(r["sample_idx"]): base64.b64decode(r["original_text_b64"]).decode("utf-8") for r in csv.DictReader(f)}
    with open(out_b / "raw_results.csv", "r", encoding="utf-8") as f:
        rows_b = {int(r["sample_idx"]): base64.b64decode(r["original_text_b64"]).decode("utf-8") for r in csv.DictReader(f)}

    # Detector A kept samples 0 and 2; Detector B kept samples 1 and 2
    assert list(rows_a.keys()) == [0, 2]
    assert list(rows_b.keys()) == [1, 2]

    # Sample #2 is the exact same text across both detectors
    assert rows_a[2] == rows_b[2]


def test_parse_args_greedy_and_non_greedy_flags():
    """Verify CLI argument parsing for default greedy mode and --non-greedy legacy flag."""
    with patch("sys.argv", ["run_analysis.py"]):
        args = parse_args()
        assert args.greedy is True
        assert args.non_greedy is False

    with patch("sys.argv", ["run_analysis.py", "--non-greedy"]):
        args = parse_args()
        assert args.greedy is False
        assert args.non_greedy is True

    with patch("sys.argv", ["run_analysis.py", "--no-greedy"]):
        args = parse_args()
        assert args.greedy is False
        assert args.non_greedy is True

    with patch("sys.argv", ["run_analysis.py", "--greedy"]):
        args = parse_args()
        assert args.greedy is True
        assert args.non_greedy is False

    with patch("sys.argv", ["run_analysis.py", "--non-greedy", "--greedy"]):
        args = parse_args()
        assert args.greedy is True
        assert args.non_greedy is False

    with patch("sys.argv", ["run_analysis.py", "--greedy", "--non-greedy"]):
        args = parse_args()
        assert args.greedy is False
        assert args.non_greedy is True


def test_greedy_mode_uses_previous_iteration_as_base_and_accumulates_budget(tmp_path):
    """
    Verify that in default greedy mode:
    1. Iteration k uses iteration (k-1)'s perturbed text as its input base.
    2. Iteration k only receives the incremental budget delta (e.g. 0.01 at 1%, 0.01 at 2%, 0.01 at 3%).
    3. raw_results.csv preserves original_text_b64 as the unperturbed text while recording cumulative
       real_budget_consumed and progressive perturbed_text_b64 at each budget step.
    """
    import base64

    orig_text = " ".join([f"word{i}" for i in range(100)])
    data_file = tmp_path / "texts.json"
    data_file.write_text(json.dumps([{"text": orig_text, "label": 1}]))

    output_dir = tmp_path / "run_greedy_default"
    attack_calls = []

    def mock_attack_text(text, budget_frac, attacker):
        step_num = len(attack_calls) + 1
        attack_calls.append((text, budget_frac))
        return f"{text} [edit_{step_num}]", 1, True

    mock_detector = MagicMock()
    mock_detector.predict.side_effect = lambda texts: [1] * len(texts)
    mock_detector.predict_proba.side_effect = lambda texts: [[0.2, 0.8]] * len(texts)

    with patch("run_analysis.load_detector") as mock_load_det, \
         patch("run_analysis.attack_text", side_effect=mock_attack_text):

        mock_load_det.return_value = (
            mock_detector,
            "mock_det",
            {"display_name": "Mock Detector"},
            Path("/mock/model"),
        )

        res_dir = run_single_analysis(
            detector_type="logistic_regression",
            attacker_type="zero_width",
            num_samples=1,
            max_budget=0.03,
            budget_steps=3,
            seed=42,
            data_path=data_file,
            output_dir=output_dir,
        )

    assert res_dir is not None
    assert len(attack_calls) == 3

    # Step 1 (1%): base is orig_text, step budget is 0.01
    assert attack_calls[0] == (orig_text, 0.01)
    # Step 2 (2%): base is Step 1's output, step budget is incremental 0.01 (NOT 0.02 from scratch)
    assert attack_calls[1] == (f"{orig_text} [edit_1]", 0.01)
    # Step 3 (3%): base is Step 2's output, step budget is incremental 0.01 (NOT 0.03 from scratch)
    assert attack_calls[2] == (f"{orig_text} [edit_1] [edit_2]", 0.01)

    with open(output_dir / "raw_results.csv", "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3

    # Verify cumulative budget consumption and base64 texts
    for idx, row in enumerate(rows, start=1):
        decoded_orig = base64.b64decode(row["original_text_b64"]).decode("utf-8")
        decoded_pert = base64.b64decode(row["perturbed_text_b64"]).decode("utf-8")
        assert decoded_orig == orig_text
        expected_edits = " ".join([f"[edit_{k}]" for k in range(1, idx + 1)])
        assert decoded_pert == f"{orig_text} {expected_edits}"
        assert int(row["real_budget_consumed"]) == idx

    cfg = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    assert cfg["greedy"] is True
    assert cfg["non_greedy"] is False


def test_non_greedy_flag_reverts_to_legacy_from_scratch_behavior(tmp_path):
    """
    Verify that passing non_greedy=True (or greedy=False) reverts to the legacy
    from-scratch behavior where every budget step starts from original_text
    with the full cumulative budget fraction.
    """
    orig_text = " ".join([f"word{i}" for i in range(100)])
    data_file = tmp_path / "texts.json"
    data_file.write_text(json.dumps([{"text": orig_text, "label": 1}]))

    output_dir = tmp_path / "run_non_greedy"
    attack_calls = []

    def mock_attack_text(text, budget_frac, attacker):
        step_num = len(attack_calls) + 1
        attack_calls.append((text, budget_frac))
        return f"{text} [scratch_{step_num}]", step_num, True

    mock_detector = MagicMock()
    mock_detector.predict.side_effect = lambda texts: [1] * len(texts)
    mock_detector.predict_proba.side_effect = lambda texts: [[0.2, 0.8]] * len(texts)

    with patch("run_analysis.load_detector") as mock_load_det, \
         patch("run_analysis.attack_text", side_effect=mock_attack_text):

        mock_load_det.return_value = (
            mock_detector,
            "mock_det",
            {"display_name": "Mock Detector"},
            Path("/mock/model"),
        )

        res_dir = run_single_analysis(
            detector_type="logistic_regression",
            attacker_type="zero_width",
            num_samples=1,
            max_budget=0.03,
            budget_steps=3,
            seed=42,
            data_path=data_file,
            output_dir=output_dir,
            non_greedy=True,
        )

    assert res_dir is not None
    assert len(attack_calls) == 3

    # Every step starts from orig_text with full cumulative budget (0.01, 0.02, 0.03)
    assert attack_calls[0] == (orig_text, 0.01)
    assert attack_calls[1] == (orig_text, 0.02)
    assert attack_calls[2] == (orig_text, 0.03)

    cfg = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    assert cfg["greedy"] is False
    assert cfg["non_greedy"] is True


def test_greedy_vs_non_greedy_real_attacker_equivalence_and_oracle_query_reduction(tmp_path):
    """
    End-to-end integration test running the REAL attack_text and REAL ZeroWidthInjectionAttack
    across 4 budget steps (5%, 10%, 15%, 20% on a 20-word email -> 1, 2, 3, 4 edits).
    Verifies that:
    1. Greedy mode produces the EXACT same perturbed texts, predictions, confidences,
       and consumed budgets at every budget step as Non-Greedy mode.
    2. Greedy mode drastically reduces the number of oracle queries by reusing the
       previous step's output and word ranking and only evaluating new candidate words.
    """
    import base64

    # 20 non-stopword-rich tokens so 5%, 10%, 15%, 20% map to 1, 2, 3, 4 edits
    sample_text = (
        "Urgent security alert please verify corporate account password "
        "credentials immediately avoid suspension banking access portal "
        "login confirmation required today now"
    )
    assert len(sample_text.split()) == 20

    data_file = tmp_path / "texts.json"
    data_file.write_text(json.dumps([{"text": sample_text, "label": 1}]))

    sensitive_roots = ["password", "credentials", "verify", "account", "urgent", "security"]

    def make_counting_detector():
        det = MagicMock()
        query_count = [0]

        def score_text(t: str) -> float:
            # Each unsplit sensitive word contributes to phishing confidence
            words_lower = [w.lower() for w in t.split()]
            hits = sum(1 for root in sensitive_roots if root in words_lower)
            # Base 0.55 + 0.07 per unperturbed sensitive keyword
            return round(min(0.97, 0.55 + 0.07 * hits), 4)

        def predict(texts):
            query_count[0] += len(texts)
            return [1 if score_text(t) >= 0.50 else 0 for t in texts]

        def predict_proba(texts):
            res = []
            for t in texts:
                p1 = score_text(t)
                res.append([round(1.0 - p1, 4), p1])
            return res

        det.predict.side_effect = predict
        det.predict_proba.side_effect = predict_proba
        return det, query_count

    det_greedy, count_greedy = make_counting_detector()
    det_non_greedy, count_non_greedy = make_counting_detector()

    out_greedy = tmp_path / "real_greedy"
    out_non_greedy = tmp_path / "real_non_greedy"

    with patch("run_analysis.load_detector", return_value=(det_greedy, "lr", {"display_name": "LR"}, Path("/mock/lr"))):
        run_single_analysis(
            detector_type="logistic_regression",
            attacker_type="zero_width",
            num_samples=1,
            max_budget=0.20,
            budget_steps=4,
            seed=42,
            data_path=data_file,
            output_dir=out_greedy,
            greedy=True,
        )

    with patch("run_analysis.load_detector", return_value=(det_non_greedy, "lr", {"display_name": "LR"}, Path("/mock/lr"))):
        run_single_analysis(
            detector_type="logistic_regression",
            attacker_type="zero_width",
            num_samples=1,
            max_budget=0.20,
            budget_steps=4,
            seed=42,
            data_path=data_file,
            output_dir=out_non_greedy,
            non_greedy=True,
        )

    with open(out_greedy / "raw_results.csv", "r", encoding="utf-8") as f:
        rows_greedy = list(csv.DictReader(f))
    with open(out_non_greedy / "raw_results.csv", "r", encoding="utf-8") as f:
        rows_non_greedy = list(csv.DictReader(f))

    assert len(rows_greedy) == 4
    assert len(rows_non_greedy) == 4

    for rg, rng in zip(rows_greedy, rows_non_greedy):
        assert rg["budget"] == rng["budget"]
        pert_g = base64.b64decode(rg["perturbed_text_b64"]).decode("utf-8")
        pert_ng = base64.b64decode(rng["perturbed_text_b64"]).decode("utf-8")
        assert pert_g == pert_ng
        assert rg["perturbed_confidence"] == rng["perturbed_confidence"]
        assert rg["real_budget_consumed"] == rng["real_budget_consumed"]
        assert rg["transformation_success"] == rng["transformation_success"] == "True"

    # Greedy mode must make less than half as many detector/oracle queries as Non-Greedy mode
    assert count_greedy[0] < count_non_greedy[0] * 0.5


def test_greedy_mode_stops_querying_oracle_after_evasion_or_exhaustion(tmp_path):
    """
    Verify that in greedy mode (even with skip_flipped=False), once a sample flips
    detection at Step 1 (or exhausts all candidate edits), Steps 2 and 3 reuse the
    perturbed result without making redundant oracle calls or flipping a benign text
    back to phishing, while preserving transformation_success=True and consumed budget.
    """
    sample_text = (
        "Urgent security alert please verify corporate account password "
        "credentials immediately avoid suspension banking access portal "
        "login confirmation required today now"
    )
    data_file = tmp_path / "texts.json"
    data_file.write_text(json.dumps([{"text": sample_text, "label": 1}]))

    det = MagicMock()
    oracle_queries = []

    def score_text(t: str) -> float:
        # As soon as "support@microsoft.com" is injected by SenderHeaderAttack, flip to benign (0.15)
        if "support@microsoft.com" in t:
            return 0.15
        return 0.92

    def predict(texts):
        for t in texts:
            oracle_queries.append(t)
        return [1 if score_text(t) >= 0.50 else 0 for t in texts]

    def predict_proba(texts):
        return [[round(1.0 - score_text(t), 4), score_text(t)] for t in texts]

    det.predict.side_effect = predict
    det.predict_proba.side_effect = predict_proba

    out_dir = tmp_path / "greedy_early_evade"
    with patch("run_analysis.load_detector", return_value=(det, "lr", {"display_name": "LR"}, Path("/mock/lr"))):
        run_single_analysis(
            detector_type="logistic_regression",
            attacker_type="sender_header",
            num_samples=1,
            max_budget=0.15,
            budget_steps=3,
            seed=42,
            data_path=data_file,
            output_dir=out_dir,
            skip_flipped=False,
            greedy=True,
        )

    with open(out_dir / "raw_results.csv", "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 3
    for r in rows:
        assert r["evaded"] == "1"
        assert r["perturbed_pred"] == "0"
        assert r["transformation_success"] == "True"
        assert int(r["real_budget_consumed"]) > 0

    # Oracle queries:
    # 1 initial predict + (1 original check + 1 first candidate flip) in Step 1 + 3 step batch predicts = 6 total!
    # Steps 2 and 3 made 0 attacker oracle queries because Step 1 already evaded.
    assert len(oracle_queries) == 6



