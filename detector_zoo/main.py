import argparse
import os
from src.data_loader import load_data
from src.splitter import split_data
from src.model_pipeline import create_baseline_pipeline
from src.metrics import evaluate_model, generate_report

def main():
    parser = argparse.ArgumentParser(description="Run the baseline evasion engine model.")
    parser.add_argument("--data", type=str, default="combined_full.json", help="Path to the JSON dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split size.")
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    try:
        df = load_data(args.data)
        print(f"Loaded {len(df)} records.")
    except FileNotFoundError:
        print(f"Error: Dataset not found at {args.data}")
        return

    # To handle potential memory constraints, if dataset is massive, we can subset it here for the baseline
    # But as per requirements we use it as is if it fits in memory.
    # We drop any NaNs just in case
    df = df.dropna(subset=['text', 'label'])
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df, test_size=args.test_size, random_state=args.seed)
    
    print("Building model pipeline...")
    pipeline = create_baseline_pipeline(random_state=args.seed)
    
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)
    
    seeds_info = {
        'split_random_state': args.seed,
        'model_random_state': args.seed,
        'test_size': args.test_size,
        'dataset_size': len(df)
    }
    
    report = generate_report(metrics, seeds_info)
    
    print("\n" + report)
    
    # Save report
    os.makedirs('reports', exist_ok=True)
    report_path = 'reports/baseline_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    os.makedirs('models', exist_ok=True)
    model_path = 'models/logistic_regression'
    pipeline.save_model(model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
