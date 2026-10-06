import sys
import json
import argparse
from src.model_pipeline import BaselineDetector
from src.char_cnn import CharCNNDetector
from src.transformer_detector import TransformerDetector

# Maps the --type flag to the correct detector class and default model path.
DETECTOR_MAP = {
    "baseline": {
        "class": BaselineDetector,
        "default_model": "models/logistic_regression",
    },
    "charcnn": {
        "class": CharCNNDetector,
        "default_model": "models/char_cnn",
    },
    "transformer": {
        "class": TransformerDetector,
        "default_model": "models/transformer",
    },
}

LABELS = {0: "benign", 1: "phishing"}


def main():
    parser = argparse.ArgumentParser(
        description="Predict using a trained detector from the zoo."
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input text file.")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to the trained model (overrides --type default).")
    parser.add_argument(
        "--type", type=str, default="baseline",
        choices=list(DETECTOR_MAP.keys()),
        help="Detector type to use. Default: baseline.",
    )
    parser.add_argument("--json", action="store_true", dest="json_output",
                        help="Output results as JSON.")
    args = parser.parse_args()

    entry = DETECTOR_MAP[args.type]
    DetectorCls = entry["class"]
    model_path = args.model or entry["default_model"]

    try:
        model = DetectorCls.load_model(model_path)
    except FileNotFoundError:
        print(
            f"Error: Model not found at {model_path}. "
            f"Please train the '{args.type}' model first.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    text = ""
    if args.input:
        try:
            with open(args.input, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            print(f"Error: Input file not found at {args.input}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error reading input file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Read from stdin
        if not sys.stdin.isatty():
            text = sys.stdin.read()
        else:
            print("Please provide input via stdin or --input flag.", file=sys.stderr)
            sys.exit(1)

    if not text.strip():
        print("No input text provided.", file=sys.stderr)
        sys.exit(1)

    probs = model.predict_proba([text])[0]       # [P(benign), P(phishing)]
    prediction = int(probs.argmax())
    confidence = float(probs[prediction])
    label = LABELS[prediction]

    if args.json_output:
        result = {
            "prediction": prediction,
            "label": label,
            "confidence": round(confidence, 4),
            "probabilities": {
                "benign": round(float(probs[0]), 4),
                "phishing": round(float(probs[1]), 4),
            },
            "detector": args.type,
            "model": model_path,
        }
        print(json.dumps(result, indent=2))
    else:
        print(f"Prediction : {prediction} ({label})")
        print(f"Confidence : {confidence:.2%}")
        print(f"P(benign)  : {probs[0]:.4f}")
        print(f"P(phishing): {probs[1]:.4f}")
        print(f"Detector   : {args.type}")


if __name__ == "__main__":
    main()

