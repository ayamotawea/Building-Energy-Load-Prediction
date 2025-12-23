import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings('ignore')


dagshub.init(repo_owner='ayamotawea', repo_name='test_daghub', mlflow=True)

def train_xgboost(
    X_train, X_test,
    y_train, y_test,
    plot_name,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8
):


    mlflow.set_experiment('building-energy-prediction-DagsHub')

    with mlflow.start_run():

        mlflow.set_tag("model", "XGBoost_MultiOutput")

        # ---- Model ----
        clf = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                objective="reg:squarederror",
                random_state=45,
                n_jobs=-1
            )
        )

        clf.fit(X_train, y_train)

        # ---- Predictions ----
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)

        # ---- Metrics ----
        mse_train = mean_squared_error(y_train, y_pred_train, multioutput="raw_values")
        mse_test = mean_squared_error(y_test, y_pred_test, multioutput="raw_values")

        mae_train = mean_absolute_error(y_train, y_pred_train, multioutput="raw_values")
        mae_test = mean_absolute_error(y_test, y_pred_test, multioutput="raw_values")

        r2_train = r2_score(y_train, y_pred_train, multioutput="raw_values")
        r2_test = r2_score(y_test, y_pred_test, multioutput="raw_values")

        # ---- Log params ----
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree
        })

        # ---- Log metrics ----
        mlflow.log_metrics({
            "Total_MSE_train": mse_train.mean(),
            "Total_MSE_test": mse_test.mean(),
            "Total_MAE_train": mae_train.mean(),
            "Total_MAE_test": mae_test.mean(),
            "Total_R2_train": r2_train.mean(),
            "Total_R2_test": r2_test.mean(),

            "Output1_R2_test": r2_test[0],
            "Output2_R2_test": r2_test[1]
        })

        # ---- Log model ----
        mlflow.sklearn.log_model(
            clf,
            artifact_path=f"model/{plot_name}"
        )

        # ---- Feature Importance ----
        feature_names = (
            X_train.columns.tolist()
            if hasattr(X_train, "columns")
            else [f"Feature {i}" for i in range(X_train.shape[1])]
        )

        targets = ["Output 1", "Output 2"]

        for i, target in enumerate(targets):

            importances = clf.estimators_[i].feature_importances_
            indices = np.argsort(importances)[::-1]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(
                np.array(feature_names)[indices],
                importances[indices]
            )
            ax.set_title(f"XGBoost Feature Importance - {target}")
            ax.invert_yaxis()

            mlflow.log_figure(
                fig,
                f"{plot_name}_xgb_feature_importance_{target}.png"
            )

            plt.close(fig)

        return clf
    
    

if __name__ == '__main__':
    
    X_train=pd.read_csv('Building-Energy-Load-Prediction/data/X_train.csv')
    y_train=pd.read_csv('Building-Energy-Load-Prediction/data/X_train.csv')
    X_test=pd.read_csv('Building-Energy-Load-Prediction/data/X_train.csv')
    y_test=pd.read_csv('Building-Energy-Load-Prediction/data/X_train.csv')
    
    train_model(X_train,X_test, y_train,y_test,'all_features',n_estimators=150, learning_rate=0.08, max_depth=10)
    train_model(X_train,X_test, y_train,y_test,'all_features',n_estimators=175, learning_rate=0.2, max_depth=5,colsample_bytree= 1,subsample= 0.8)
    train_model(X_train,X_test, y_train,y_test,'all_features',reg_lambda=1.0,reg_alpha=0.5,n_estimators=175, learning_rate=0.2, max_depth=5,colsample_bytree= 1,subsample= 0.8)

