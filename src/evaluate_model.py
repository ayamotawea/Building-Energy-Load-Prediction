import argparse
import joblib
import json
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

FEATURES = [
    "Relative Compactness", "Surface Area", "Wall Area", "Roof Area",
    "Overall Height", "Orientation", "Glazing Area", "Glazing Area Distribution"
]
TARGETS = ["HL", "CL"]

def evaluate(data_path, model_path, target):
    df = pd.read_csv(data_path)
    X = df[FEATURES].copy()
    y = df[target].values

    model = joblib.load(model_path)
    y_pred = model.predict(X)

    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    metrics = {"mae": float(mae), "mse": float(mse), "r2": float(r2)}
    print(json.dumps(metrics, indent=2))

    plt.figure()
    plt.scatter(y, y_pred, alpha=0.7)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Actual vs Predicted — {target} (Full Dataset)")
    plt.tight_layout()
    out_png = model_path.replace(".pkl", "_eval.png")
    plt.savefig(out_png, dpi=200)
    plt.close()
    return metrics, out_png

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to CSV dataset")
    parser.add_argument("--model_path", required=True, help="Path to trained model .pkl")
    parser.add_argument("--target", required=True, choices=TARGETS, help="Target to evaluate (HL or CL)")
    args = parser.parse_args()
    evaluate(args.data_path, args.model_path, args.target)