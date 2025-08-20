import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

FEATURES = [
    "Relative Compactness", "Surface Area", "Wall Area", "Roof Area",
    "Overall Height", "Orientation", "Glazing Area", "Glazing Area Distribution"
]
TARGETS = ["HL", "CL"]

def load_data(path):
    df = pd.read_csv(path)
    # Normalize column names to handle variants
    cols = {c.strip(): c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)
    return df

def train_model(data_path, target, outdir, test_size=0.15, random_state=42):
    assert target in TARGETS, f"target must be one of {TARGETS}"
    os.makedirs(outdir, exist_ok=True)

    df = load_data(data_path)
    X = df[FEATURES].copy()
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("xgb", XGBRegressor(objective="reg:squarederror", random_state=random_state))
    ])

    param_grid = {
        "xgb__n_estimators": [150, 175, 200],
        "xgb__learning_rate": [0.1, 0.2],
        "xgb__max_depth": [4, 5, 6],
        "xgb__subsample": [0.8, 1.0],
        "xgb__colsample_bytree": [0.8, 1.0],
        "xgb__reg_lambda": [1.0],
        "xgb__reg_alpha": [0.0, 0.5],
    }

    gs = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=-1, scoring="neg_mean_absolute_error", verbose=0)
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    metrics = {
        "train": {
            "mae": float(mean_absolute_error(y_train, y_pred_train)),
            "mse": float(mean_squared_error(y_train, y_pred_train)),
            "r2": float(r2_score(y_train, y_pred_train)),
        },
        "test": {
            "mae": float(mean_absolute_error(y_test, y_pred_test)),
            "mse": float(mean_squared_error(y_test, y_pred_test)),
            "r2": float(r2_score(y_test, y_pred_test)),
        },
        "best_params": gs.best_params_,
    }

    # Save model and metrics
    joblib.dump(best_model, os.path.join(outdir, "model.pkl"))
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        import json
        json.dump(metrics, f, indent=2)

    # Feature importance plot (gain-based) if supported
    try:
        reg = best_model.named_steps["xgb"]
        importances = reg.feature_importances_
        plt.figure()
        plt.bar(range(len(FEATURES)), importances)
        plt.xticks(range(len(FEATURES)), FEATURES, rotation=45, ha="right")
        plt.title(f"Feature Importance — {target}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "feature_importance.png"), dpi=200)
        plt.close()
    except Exception as e:
        print("Could not produce feature importance plot:", e)

    # Predicted vs Actual plot
    plt.figure()
    plt.scatter(y_test, y_pred_test, alpha=0.7)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Actual vs Predicted — {target}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "actual_vs_pred.png"), dpi=200)
    plt.close()

    print("Training complete. Metrics:")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to CSV dataset")
    parser.add_argument("--target", required=True, choices=TARGETS, help="Target to train (HL or CL)")
    parser.add_argument("--outdir", default="models", help="Output directory")
    args = parser.parse_args()
    train_model(args.data_path, args.target, args.outdir)