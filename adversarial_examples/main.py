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
    parser.add_argument("--budget", type=float, default=1.0, help="Maximum percentage of text to perturb (e.g. 0.5 for 50%).")
    
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
        
    # Redirect stdout and stderr temporarily to ensure ONLY the output sentence is printed
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    
    try:
        from harness.budget import PerturbationBudget
        from harness.content_attacks import (
            DynamicSynonymSubstitution
        )
        
        # Compute max_edits based on the budget percentage and number of words
        word_count = len(text.split())
        max_edits = max(1, int(word_count * args.budget)) if args.budget > 0 else 0
        budget = PerturbationBudget(max_edits=max_edits)
        
        # Apply sequential transformations
        transformers = [
            DynamicSynonymSubstitution()
        ]
        
        current_text = text
        for transformer in transformers:
            try:
                current_text = transformer.transform(current_text, budget)
            except Exception:
                # If budget is exceeded or other transformation error occurs, just skip to the next
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
