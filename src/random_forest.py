from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import dagshub
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


dagshub.init(repo_owner='ayamotawea', repo_name='Building-Energy-Load-Prediction', mlflow=True)

def train_model(X_train, X_test, y_train, y_test, plot_name):

    mlflow.set_experiment('building-energy-prediction-DagsHub')

    with mlflow.start_run():

        mlflow.set_tag('model', 'RandomForest_MultiOutput')

        # ---- Base model ----
        base_model = MultiOutputRegressor(
            RandomForestRegressor(random_state=45)
        )

        # ---- Grid Search ----
        param_grid = {
            'estimator__n_estimators': [100, 150, 200],
            'estimator__max_depth': [8, 12, 16],
            'estimator__min_samples_split': [2, 5],
            'estimator__min_samples_leaf': [1, 2]
        }

        grid = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        # ---- Best model ----
        clf = grid.best_estimator_

        # ---- Log best params ----
        mlflow.log_params(grid.best_params_)

        # ---- Predictions ----
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)

        # ---- Metrics (Train) ----
        mse_train = mean_squared_error(y_train, y_pred_train, multioutput='raw_values')
        mae_train = mean_absolute_error(y_train, y_pred_train, multioutput='raw_values')
        r2_train = r2_score(y_train, y_pred_train, multioutput='raw_values')

        # ---- Metrics (Test) ----
        mse_test = mean_squared_error(y_test, y_pred_test, multioutput='raw_values')
        mae_test = mean_absolute_error(y_test, y_pred_test, multioutput='raw_values')
        r2_test = r2_score(y_test, y_pred_test, multioutput='raw_values')

        # ---- Log metrics ----
        mlflow.log_metrics({
            "Total_MAE_train": mae_train.mean(),
            "Total_MAE_test": mae_test.mean(),
            "Total_MSE_train": mse_train.mean(),
            "Total_MSE_test": mse_test.mean(),
            "Total_R2_train": r2_train.mean(),
            "Total_R2_test": r2_test.mean(),

            "Output1_R2_test": r2_test[0],
            "Output2_R2_test": r2_test[1],
        })

        # ---- Log model ----
        mlflow.sklearn.log_model(
            clf,
            artifact_path=f"model/{plot_name}"
        )

        # ---- Feature Importance ----
        feature_importances = np.array([
            est.feature_importances_ for est in clf.estimators_
        ])

        feature_names = (
            X_train.columns.tolist()
            if hasattr(X_train, "columns")
            else [f"Feature {i}" for i in range(X_train.shape[1])]
        )

        targets = ["Output 1", "Output 2"]

        for i, target in enumerate(targets):

            indices = np.argsort(feature_importances[i])[::-1]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(
                np.array(feature_names)[indices],
                feature_importances[i][indices]
            )
            ax.set_title(f"Feature Importance - {target}")
            ax.invert_yaxis()

            mlflow.log_figure(
                fig,
                f"{plot_name}_feature_importance_{target}.png"
            )
            plt.close(fig)

        return clf

        




if __name__ == '__main__':
    
    X_train=pd.read_csv('Building-Energy-Load-Prediction/data/X_train.csv')
    y_train=pd.read_csv('Building-Energy-Load-Prediction/data/X_train.csv')
    X_test=pd.read_csv('Building-Energy-Load-Prediction/data/X_train.csv')
    y_test=pd.read_csv('Building-Energy-Load-Prediction/data/X_train.csv')

    train_model(X_train,X_test, y_train,y_test,'all_features')