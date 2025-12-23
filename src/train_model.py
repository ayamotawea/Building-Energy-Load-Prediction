import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from pathlib import Path

# Get project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define directories relative to project root
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
df = pd.read_csv(f"{PROJECT_ROOT}/ENB2012_data.csv")

X = df[['X1', 'X2', 'X3', 'X4', 'X5','X6', 'X7', 'X8']]
y = df[['Y1', 'Y2']] 


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.15, random_state=42)


scaler=StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_df = pd.DataFrame(X_train)
X_test_df = pd.DataFrame(X_test)
y_train=pd.DataFrame(y_train)
y_test=pd.DataFrame(y_test)



X_train_df.to_csv(os.path.join(DATA_DIR, 'X_train_processed.csv'), index=False)
X_test_df.to_csv(os.path.join(DATA_DIR, 'X_test_processed.csv'), index=False)
y_train.to_csv(os.path.join(DATA_DIR, 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(DATA_DIR, 'y_test.csv'), index=False)


model= MultiOutputRegressor(XGBRegressor(reg_lambda=1.0,reg_alpha=0.5,n_estimators=175, learning_rate=0.2, max_depth=5,colsample_bytree= 1,subsample= 0.8 ,random_state=42))
model.fit(X_train, y_train)




joblib.dump(model, f'{MODELS_DIR}/model.pkl')
joblib.dump(scaler, f'{MODELS_DIR}/scaler.pkl')
