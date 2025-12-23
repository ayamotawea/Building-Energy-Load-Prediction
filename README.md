
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
- 📑 Full report available in [`docs/report.pdf`](docs/report.pdf)

---

## 📂 Project Structure
```
Building-Energy-Load-Prediction/
│
├── 📁 notebook/
│   └── Preprocessing_EDA_data.ipynb # EDA and preprocessing exploration only
│
├── 📁 src/
│   ├── train_model.py        # Train XGBoost models for HL & CL
│   ├── evaluate_model.py     # Evaluate trained models on any dataset
│   ├── random_forest.py      # Optional alternative model
│   └── xgboost_model.py      # XGBoost implementation
│
├── 📁 docs/
│   └── report.pdf            # Full project report with diagrams & results
│
├── 📁 data/                  # Processed datasets (ignored in GitHub)
├── 📁 models/                # Saved trained model and scaler (ignored in GitHub)
├── ENB2012_data.csv          # Raw Kaggle dataset
├── BuildingEnergy.py         # Pydantic models for FastAPI responses
├── app.py                    # FastAPI app to serve predictions
├── requirements.txt          # Dependencies
├── README.md                 # Project documentation
└── .gitignore                # Ignore unnecessary files & folders
```

---

## 🧩 Dataset
The dataset comes from **Kaggle**:  
🔗 [Energy Efficiency Dataset](https://www.kaggle.com/datasets/elikplim/eergy-efficiency-dataset)

- **768 residential buildings**
- **8 input features**
- **2 outputs:** Heating Load (HL) & Cooling Load (CL)

> **Note:** `ENB2012_data.csv` is included, but processed datasets are generated via the training script.

---

## 🛠 Installation
```bash
git clone https://github.com/ayamotawea/Building-Energy-Load-Prediction.git
cd Building-Energy-Load-Prediction
pip install -r requirements.txt
```

---

## 🚀 How to Run the Project

### **Step 1 — Explore Data (Optional)**
```bash
jupyter notebook notebook/Preprocessing_EDA_data.ipynb
```
- Perform EDA & preprocessing to understand data patterns
- This notebook is only for exploration; training should be done via script

---

### **Step 2 — Train Model**

```bash
python src/train_model.py
```
- Trains XGBoost models for Heating Load (HL) and Cooling Load (CL)
- Saves processed datasets in data/ and trained model & scaler in models/
- Fully reproducible and suitable for production

---

### **Step 3 — Evaluate Model (Optional)**

```bash
python src/evaluate_model.py --save_metrics

```
- Loads trained model and processed datasets
- Computes MAE, MSE, R² for train and test sets
- Prints metrics to terminal and optionally saves them to CSV

---
### **Step 4 — Run FastAPI for test**

```bash
uvicorn app:app --reload
```
- Use POST request to /predict with JSON input (see BuildingEnergy.py for schema)
- Receive predictions as JSON response

```bash
Example input:
{
    "Relative_Compactness": 0.8,
    "Surface_Area": 600,
    "Wall_Area": 300,
    "Roof_Area": 200,
    "Overall_Height": 7,
    "Orientation": 2,
    "Glazing_Area": 0.1,
    "Glazing_Area_Distribution": 3
}

```
Example response:
```bash
{
    "Heating_Load": 15.23,
    "Cooling_Load": 22.14
}

```
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
AI Engineer | ML, DL, CV, GenAI Specialist 
📧 Email: Aya.Motawea.AI@gmail.com 

🔗 [LinkedIn](https://www.linkedin.com/in/aya-motawea-661633251/) 

💻 [GitHub](https://github.com/ayamotawea)
