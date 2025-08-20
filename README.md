# 🏠 Building Energy Load Prediction using XGBoost

This repository develops a **statistical machine learning framework** to predict **Heating Load (HL)** and **Cooling Load (CL)** of residential buildings using **eight architectural features** (Relative Compactness, Surface Area, Wall Area, Roof Area, Overall Height, Orientation, Glazing Area, Glazing Area Distribution).

We conduct classical + non-parametric analysis for feature relationships and train **XGBoost** models for HL & CL with strong performance (R² ≈ 0.99 on test). The project includes a polished notebook, runnable Python scripts, and a full report with diagrams.

---

## 📌 Highlights
- Feature analysis (correlation heatmap, pair plots, outliers)
- XGBoost regression for **HL** and **CL**
- GridSearchCV-based hyperparameter tuning
- Metrics: **MAE**, **MSE**, **R²**
- Feature importance visualization
- Ready-to-run scripts (no need for Jupyter)

---

## 🧩 Dataset
**Energy Efficiency Dataset (Kaggle)**  
Link: https://www.kaggle.com/datasets/elikplim/energy-efficiency-dataset

> Note: The dataset is **not** committed to this repo. Download it from Kaggle and place the CSV at a convenient path (e.g., `data/ENB2012_data.csv`).

---

## 🛠️ Environment Setup
```bash
git clone https://github.com/yourusername/Building-Energy-Load-Prediction.git
cd Building-Energy-Load-Prediction
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 1) Run the Jupyter Notebook
Open the notebook and run all cells:
```
jupyter notebook notebook/energy-consumption-in-smart-cities.ipynb
```

### 2) Run via Python Scripts (no notebook)
Train a model and save artifacts:
```bash
python src/train_model.py --data_path data/ENB2012_data.csv --target HL --outdir models_hl
python src/train_model.py --data_path data/ENB2012_data.csv --target CL --outdir models_cl
```

Evaluate a saved model:
```bash
python src/evaluate_model.py --data_path data/ENB2012_data.csv --model_path models_hl/model.pkl --target HL
python src/evaluate_model.py --data_path data/ENB2012_data.csv --model_path models_cl/model.pkl --target CL
```

---

## 📊 Example Results (from our experiments)
| Metric | Train (HL) | Train (CL) | Test (HL) | Test (CL) |
|--------|------------|------------|-----------|-----------|
| MAE    | 0.1213     | 0.1993     | 0.2501    | 0.4843    |
| MSE    | 0.0272     | 0.0780     | 0.1250    | 0.5881    |
| R²     | 0.9997     | 0.9991     | 0.9988    | 0.9934    |

> These values were obtained on an 85/15 train/test split using tuned XGBoost models.

---

## 🔍 Feature Importance (Key Insights)
- **Relative Compactness** → most influential for both HL & CL
- **Glazing Area** → second most important
- **Wall/Surface Area** → minor but relevant
- **Orientation / Roof Area** → limited contribution

---

## 📄 Full Report (with Diagrams)
See `docs/paper&diagrams.pdf` for the complete methodology, figures, and discussion.

---

## 📬 Contact
**Aya Alaa Abd Elsalam Motwea**  
AI Engineer | Machine Learning Researcher  
Email: yoyomotawaa@gmail.com  
LinkedIn: https://www.linkedin.com/in/aya-motawea-661633251/  
GitHub: https://github.com/ayamotawea

---

_Updated: 2025-08-20_