
# 🏠 Building Energy Load Prediction using XGBoost

This project develops a **statistical machine learning framework** to predict **Heating Load (HL)** and **Cooling Load (CL)** of residential buildings using **eight architectural features**:

- Relative Compactness  
- Surface Area  
- Wall Area  
- Roof Area  
- Overall Height  
- Orientation  
- Glazing Area  
- Glazing Area Distribution  

We perform statistical analysis, visualize feature relationships, and use **XGBoost** for regression modeling.  
Our model achieves **R² ≈ 0.99** and **low MAE/MSE**, outperforming classical regression methods.

---

## 📌 Highlights
- 🧠 **ML Model:** XGBoost with hyperparameter tuning (GridSearchCV)
- 📊 **Evaluation Metrics:** MAE, MSE, R²  
- 📈 **Visual Insights:** Correlation heatmaps, feature importance, prediction plots  
- ⚡ **High Accuracy:** R² > 0.99 on test data  
- 📑 Full report available in [`docs/report.pdf`](docsreport.pdf)

---

## 📂 Project Structure
```
Building-Energy-Load-Prediction/
│
├── 📁 notebook/
│   └── energy-consumption-in-smart-cities.ipynb   # Jupyter notebook with analysis & training
│
├── 📁 src/
│   ├── train_model.py        # Script to train XGBoost models for HL & CL
│   ├── evaluate_model.py     # Script to evaluate trained models
│
├── 📁 docs/
│   └── report.pdf    # Full project report with diagrams & results
│
├── requirements.txt          # Required dependencies
├── README.md                # Project documentation
└── .gitignore               # Ignore unnecessary files & folders
```

---

## 🧩 Dataset
The dataset comes from **Kaggle**:  
🔗 [Energy Efficiency Dataset](https://www.kaggle.com/datasets/elikplim/eergy-efficiency-dataset)

- **768 residential buildings**
- **8 input features**
- **2 outputs:** Heating Load (HL) & Cooling Load (CL)

> **Note:** The dataset is **not uploaded** to this repository. Please download it manually from Kaggle.

---

## 🛠 Installation
```bash
git clone https://github.com/ayamotawea/Building-Energy-Load-Prediction.git
cd Building-Energy-Load-Prediction
pip install -r requirements.txt
```

---

## 🚀 How to Run the Project

### **Option 1 — Using the Jupyter Notebook**
```bash
jupyter notebook notebook/energy-consumption-in-smart-cities.ipynb
```
- Open the notebook and **Run All**.
- Produces visualizations, training results, and evaluation metrics.

---

### **Option 2 — Using Python Scripts**

#### **Step 1 — Train Models**
Since trained models are **not included** in the repo, you must **train them first**:
```bash
python src/train_model.py --data_path path_to_dataset.csv --target HL --outdir models_hl
python src/train_model.py --data_path path_to_dataset.csv --target CL --outdir models_cl
```
This will:
- Train XGBoost models for HL & CL.
- Save models (`model.pkl`), metrics, and plots in `models_hl/` and `models_cl/`.

#### **Step 2 — Evaluate Models**
After training, evaluate the models:
```bash
python src/evaluate_model.py --data_path path_to_dataset.csv --model_path models_hl/model.pkl --target HL
python src/evaluate_model.py --data_path path_to_dataset.csv --model_path models_cl/model.pkl --target CL
```
This generates:
- Performance metrics (MAE, MSE, R²)
- Actual vs Predicted plots

---

## 📊 Results

| Metric | Train (HL) | Train (CL) | Test (HL) | Test (CL) |
|--------|------------|------------|-----------|-----------|
| **MAE** | 0.1213 | 0.1993 | 0.2501 | 0.4843 |
| **MSE** | 0.0272 | 0.0780 | 0.1250 | 0.5881 |
| **R²**  | 0.9997 | 0.9991 | 0.9988 | 0.9934 |

---

## 🔍 Feature Importance (Key Insights)
- **Relative Compactness** → Most influential for both HL & CL.
- **Glazing Area** → Second most significant factor.
- **Wall & Surface Area** → Moderate contribution.
- **Orientation & Roof Area** → Minimal effect.

---

## 📄 Full Report
For a complete explanation of the methodology, results, and diagrams, check:  
📂 [`docs/report.pdf`](docs/report.pdf)

---

## 📬 Contact
**Aya Alaa Motwea**  
AI Engineer | Machine Learning Researcher  
📧 Email:Aya.Motawea.AI@gmail.com 
🔗 [LinkedIn](https://www.linkedin.com/in/aya-motawea-661633251/)  
💻 [GitHub](https://github.com/ayamotawea)
