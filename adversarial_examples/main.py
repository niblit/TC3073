import os
import sys
import argparse
import warnings
import logging

def main():
    # Suppress verbose warnings and logs from deep learning libraries
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings("ignore")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    
    parser = argparse.ArgumentParser(description="Generate adversarial examples.")
    parser.add_argument("--input", type=str, help="Input text file path.")
    parser.add_argument("--budget", type=float, default=1.0, help="Maximum percentage of text to perturb (e.g. 0.5 for 50%%).")
    parser.add_argument(
        "--oracle-cmd", type=str,
        default="cd ../detector_zoo && .venv/bin/python predict.py --type baseline --json",
        help="Shell command for the target model oracle. "
             "The command receives the text via stdin and must print "
             "a JSON object with 'label' and 'confidence' keys to stdout. "
             "Default: runs the baseline detector from ../detector_zoo."
    )
    parser.add_argument(
        "--oracle-label-key", type=str, default="label",
        help="JSON key for the predicted label in oracle output (default: 'label')."
    )
    parser.add_argument(
        "--oracle-confidence-key", type=str, default="confidence",
        help="JSON key for the confidence score in oracle output (default: 'confidence')."
    )
    
    args, _ = parser.parse_known_args()
    
    text = ""
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read().strip()
    elif not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    else:
        return
        
    if not text:
        return

    # ── Build the target model oracle ──────────────────────────────────
    def _make_subprocess_oracle(cmd, label_key, confidence_key):
        """
        Returns an oracle callable that shells out to `cmd`, pipes text
        via stdin, and parses a JSON response from stdout.
        """
        import subprocess, json

        def oracle(input_text: str):
            result = subprocess.run(
                cmd, shell=True, input=input_text,
                capture_output=True, text=True
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Oracle command failed (rc={result.returncode}): {result.stderr}"
                )
            payload = json.loads(result.stdout)
            return (str(payload[label_key]), float(payload[confidence_key]))

        return oracle

    oracle = None
    if args.oracle_cmd:
        oracle = _make_subprocess_oracle(
            args.oracle_cmd, args.oracle_label_key, args.oracle_confidence_key
        )
        
    # Redirect stdout and stderr temporarily to ensure ONLY the output sentence is printed
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    
    try:
        from harness.budget import PerturbationBudget
        from harness.content_attacks import TextFoolerAttack
        
        # Compute max_edits based on the budget percentage and number of words
        word_count = len(text.split())
        max_edits = max(1, int(word_count * args.budget)) if args.budget > 0 else 0
        budget = PerturbationBudget(max_edits=max_edits)
        
        # Apply TextFooler attack
        attack = TextFoolerAttack(target_model_oracle=oracle)
        
        current_text = text
        try:
            current_text = attack.transform(current_text, budget)
        except Exception:
            # If budget is exceeded or other transformation error occurs, just skip
            pass
                
    except Exception:
        # Fallback to the original text if something fatally fails
        current_text = text
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
    print(current_text)

if __name__ == "__main__":
    main()
