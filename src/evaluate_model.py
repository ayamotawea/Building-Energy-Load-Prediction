# src/evaluate_model.py
from pathlib import Path
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse
import os

def evaluate_model(model_path, X_train_path, y_train_path, X_test_path, y_test_path, save_metrics=False):
    # Load datasets
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)

    # Load trained model
    model = joblib.load(model_path)

    results = {}

    for dataset, X, y in [('Train', X_train, y_train), ('Test', X_test, y_test)]:
        y_pred = model.predict(X)

        mse = mean_squared_error(y, y_pred, multioutput='raw_values')
        mae = mean_absolute_error(y, y_pred, multioutput='raw_values')
        r2 = r2_score(y, y_pred, multioutput='raw_values')

        print(f"\n=== {dataset} Metrics ===")
        print(f"Total MAE: {mean_absolute_error(y, y_pred):.4f}")
        print(f"Total MSE: {mean_squared_error(y, y_pred):.4f}")
        print(f"Total R²: {r2_score(y, y_pred):.4f}")

        for i in range(len(mse)):
            print(f"Output {i+1}: MSE = {mse[i]:.4f}, MAE = {mae[i]:.4f}, R² = {r2[i]:.4f}")

        # Store metrics if needed
        results[dataset] = {
            'MSE': mse.tolist(),
            'MAE': mae.tolist(),
            'R2': r2.tolist(),
            'Total_MSE': mean_squared_error(y, y_pred),
            'Total_MAE': mean_absolute_error(y, y_pred),
            'Total_R2': r2_score(y, y_pred)
        }

    # Optionally save metrics to CSV
    if save_metrics:
        metrics_dir = Path("metrics")
        metrics_dir.mkdir(exist_ok=True)
        metrics_file = metrics_dir / "evaluation_metrics.csv"

        rows = []
        for dataset, metrics in results.items():
            for i, (m, a, r) in enumerate(zip(metrics['MSE'], metrics['MAE'], metrics['R2']), 1):
                rows.append({
                    'Dataset': dataset,
                    'Output': i,
                    'MSE': m,
                    'MAE': a,
                    'R2': r
                })
        pd.DataFrame(rows).to_csv(metrics_file, index=False)
        print(f"\nMetrics saved to {metrics_file}")



if __name__ == "__main__":
    # Get project root (parent of src/)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # Define directories relative to project root
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"

    parser = argparse.ArgumentParser(description="Evaluate trained XGBoost model")
    parser.add_argument("--model_path", type=str, default=f"{MODELS_DIR}/model.pkl", help="Path to trained model")
    parser.add_argument("--X_train", type=str, default=f"{DATA_DIR}/X_train_processed.csv", help="Path to X_train CSV")
    parser.add_argument("--y_train", type=str, default=f"{DATA_DIR}/y_train.csv", help="Path to y_train CSV")
    parser.add_argument("--X_test", type=str, default=f"{DATA_DIR}/X_test_processed.csv", help="Path to X_test CSV")
    parser.add_argument("--y_test", type=str, default=f"{DATA_DIR}/y_test.csv", help="Path to y_test CSV")
    parser.add_argument("--save_metrics", action="store_true", help="Save metrics to CSV")

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        X_train_path=args.X_train,
        y_train_path=args.y_train,
        X_test_path=args.X_test,
        y_test_path=args.y_test,
        save_metrics=args.save_metrics
    )
