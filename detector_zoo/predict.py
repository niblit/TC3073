import sys
import argparse
from src.model_pipeline import BaselineDetector

def main():
    parser = argparse.ArgumentParser(description="Predict using the trained baseline model.")
    parser.add_argument("--input", type=str, default=None, help="Path to input text file.")
    parser.add_argument("--model", type=str, default="models/logistic_regression", help="Path to the trained model.")
    args = parser.parse_args()

    model_path = args.model
    try:
        model = BaselineDetector.load_model(model_path)
    except FileNotFoundError:
        print(f"Error: Model not found at {model_path}. Please train the model first.", file=sys.stderr)
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

    prediction = model.predict([text])[0]
    
    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
