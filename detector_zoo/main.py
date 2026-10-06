import argparse
import os
from src.data_loader import load_data
from src.splitter import split_data
from src.model_pipeline import BaselineDetector
from src.char_cnn import CharCNNDetector
from src.transformer_detector import TransformerDetector
from src.metrics import evaluate_model, generate_report

# Registry of all detectors in the zoo.
# Each entry maps a short CLI name to (DetectorClass, save_directory, report_filename).
DETECTOR_ZOO = {
    "baseline": {
        "class": BaselineDetector,
        "model_dir": "models/logistic_regression",
        "report": "reports/baseline_report.txt",
        "label": "BASELINE (TF-IDF + Logistic Regression)",
    },
    "charcnn": {
        "class": CharCNNDetector,
        "model_dir": "models/char_cnn",
        "report": "reports/charcnn_report.txt",
        "label": "CHAR-CNN",
    },
    "transformer": {
        "class": TransformerDetector,
        "model_dir": "models/transformer",
        "report": "reports/transformer_report.txt",
        "label": "TRANSFORMER (DistilBERT)",
    },
}


def train_and_evaluate(detector, X_train, X_test, y_train, y_test, meta, label):
    """Train a single detector, evaluate it, and return the report string."""
    print(f"\n{'=' * 50}")
    print(f"  Training: {label}")
    print(f"{'=' * 50}")

    detector.fit(X_train, y_train)

    print(f"  Evaluating: {label}")
    y_pred = detector.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)
    report = generate_report(metrics, meta, model_name=label)

    print("\n" + report)
    return report, metrics


def main():
    parser = argparse.ArgumentParser(description="Run the detector zoo.")
    parser.add_argument("--data", type=str, default="combined_full.json",
                        help="Path to the JSON dataset.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test split size.")
    parser.add_argument(
        "--detectors", nargs="+",
        default=list(DETECTOR_ZOO.keys()),
        choices=list(DETECTOR_ZOO.keys()),
        help="Which detectors to train. Defaults to all.",
    )
    args = parser.parse_args()

    # ---- Load data ----
    print(f"Loading data from {args.data}...")
    try:
        df = load_data(args.data)
        print(f"Loaded {len(df)} records.")
    except FileNotFoundError:
        print(f"Error: Dataset not found at {args.data}")
        return

    df = df.dropna(subset=['text', 'label'])

    # ---- Split ----
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(
        df, test_size=args.test_size, random_state=args.seed,
    )

    seeds_info = {
        'split_random_state': args.seed,
        'model_random_state': args.seed,
        'test_size': args.test_size,
        'dataset_size': len(df),
    }

    os.makedirs('reports', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # ---- Train each selected detector ----
    for name in args.detectors:
        entry = DETECTOR_ZOO[name]
        DetectorCls = entry["class"]

        detector = DetectorCls(random_state=args.seed)
        report, _ = train_and_evaluate(
            detector, X_train, X_test, y_train, y_test,
            seeds_info, entry["label"],
        )

        # Save report
        with open(entry["report"], 'w') as f:
            f.write(report)
        print(f"Report saved to {entry['report']}")

        # Save model
        detector.save_model(entry["model_dir"])
        print(f"Model saved to {entry['model_dir']}")


if __name__ == "__main__":
    main()
