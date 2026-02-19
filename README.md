

# HeartIQ 🫀

A machine learning-powered web application for predicting heart disease risk using clinical and lifestyle features. HeartIQ walks through a complete ML pipeline — from raw data ingestion to a deployable Flask web app with a trained Logistic Regression model.

---

## 📁 File Structure

```
HEARTIQ/
├── .vscode/
│   └── settings.json
├── templates/
│   └── index.html              # Frontend UI for heart disease prediction
├── venv/                       # Python virtual environment
├── app.py                      # Flask web application
├── Data Set Link.txt           # Source link to the original dataset
├── heart_disease.csv           # Raw dataset (10,000 records, 21 features)
├── heart_preprocessed_final.csv# Cleaned & feature-selected dataset
├── heart_disease_model_package.pkl  # Serialized model + encoders + scaler
├── preprocess.ipynb            # Full ML pipeline notebook
└── README.md
```

---

## 📊 Dataset

- **Source:** `heart_disease.csv`
- **Size:** 10,000 rows × 21 columns
- **Target Variable:** `Heart Disease Status` (Yes / No)
- **Features include:**

| Category | Features |
|---|---|
| Demographics | Age, Gender |
| Vitals | Blood Pressure, BMI, Sleep Hours |
| Lipid Profile | Cholesterol Level, Triglyceride Level, CRP Level, Homocysteine Level |
| Binary Risk Flags | High Blood Pressure, Low HDL Cholesterol, High LDL Cholesterol, Diabetes, Family Heart Disease |
| Lifestyle | Exercise Habits, Smoking, Alcohol Consumption, Stress Level, Sugar Consumption |
| Lab Values | Fasting Blood Sugar |

---

## 🔬 ML Pipeline (`preprocess.ipynb`)

### 1. Data Loading & Exploration
- Loaded `heart_disease.csv` using Pandas
- Inspected dataset shape, column types, and value distributions

### 2. Missing Value Handling
- Numeric columns (Age, Blood Pressure, BMI, etc.) → filled with **mean** or **median**
- Categorical columns (Gender, Smoking, Diabetes, etc.) → filled with **mode**
- `Alcohol Consumption` had the most missing values (~2,586) and was imputed with its mode

### 3. Duplicate Handling
- Checked for duplicate rows → **0 duplicates found**

### 4. Outlier Handling
- Applied IQR-based outlier detection and capping on numeric features

### 5. Label Encoding
- Encoded all categorical variables using `sklearn.LabelEncoder`:
  - Gender, Exercise Habits, Smoking, Family Heart Disease, Diabetes, High Blood Pressure, Low HDL Cholesterol, High LDL Cholesterol, Alcohol Consumption, Stress Level, Sugar Consumption, Heart Disease Status

### 6. Feature Engineering
- Created 5 new derived features:
  - `Undiagnosed_Hypertension` — High BP flag without clinical diagnosis
  - `Undiagnosed_Diabetes` — High fasting blood sugar without diabetes diagnosis
  - `Hidden_Hyperlipidemia` — High LDL without cholesterol flag
  - `Stress_Sleep_Interaction` — Stress level × sleep hours interaction
  - `Inflammation_Index` — Combined CRP and homocysteine signal
  - `Age_BMI_Interaction` — Age × BMI product
  - `Lipid_Risk_Ratio` — Ratio of LDL-related risk markers
  - `Sugar_Load` — Combined sugar and fasting blood sugar signal

### 7. Train-Test Split
- 80/20 split using `train_test_split` with stratification
- Training set: **8,000 samples** | Test set: **2,000 samples**

### 8. Feature Scaling
- Applied `StandardScaler` fitted **only on training data**, then transformed the test set

### 9. SMOTE (Class Imbalance)
- Original class distribution: `0 → 6,400`, `1 → 1,600`
- After SMOTE: balanced to `0 → 6,400`, `1 → 6,400`

### 10. Feature Selection
Three methods were evaluated:
- **Filter Method** (SelectKBest / f_classif) — top 10 features
- **Wrapper Method** (RFE)
- **Embedded Method** (Random Forest Feature Importance) ← **final selection**

**Final 10 Selected Features:**
```
Exercise Habits, Alcohol Consumption, Fasting Blood Sugar, BMI,
Blood Pressure, Homocysteine Level, Cholesterol Level, Sleep Hours,
Inflammation_Index, CRP Level
```

### 11. Model Training & Evaluation

Four models were trained and compared:

| Model | Accuracy |
|---|---|
| Logistic Regression | 80.00% |
| Random Forest | 79.90% |
| Support Vector Machine | 80.00% |
| KNN Classifier | 76.00% |

> **Logistic Regression** was selected as the final model and serialized for deployment.

### 12. Model Export
- All assets saved to `heart_disease_model_package.pkl` using `joblib`:
  - Trained model (`lr_model`)
  - Scaler (`StandardScaler`)
  - Selected features list
  - All `LabelEncoder` objects per column

---

## 🌐 Web Application

A Flask web app (`app.py`) serves predictions via a browser interface:

- **Frontend:** `templates/index.html`
- **Backend:** `app.py` loads the `.pkl` package and runs inference on user input
- Users enter clinical and lifestyle data through the form and receive a risk prediction

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.13 |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML | Scikit-learn |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Model Serialization | Joblib |
| Web Framework | Flask |
| Frontend | HTML (Jinja2 templates) |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/HeartIQ.git
cd HeartIQ
```

### 2. Create & Activate Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install flask pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib
```

### 4. Run the App
```bash
python app.py
```
Then open your browser and navigate to `http://127.0.0.1:5000`

---

## 📓 Reproducing the ML Pipeline

Open and run `preprocess.ipynb` in Jupyter or VS Code. Make sure `heart_disease.csv` is in the root directory. Running all cells will reproduce the full pipeline and regenerate `heart_preprocessed_final.csv` and `heart_disease_model_package.pkl`.

---

## 📌 Notes

- The dataset link is provided in `Data Set Link.txt`
- The model was trained on a balanced dataset after SMOTE oversampling
- All preprocessing steps (scaling, encoding) are bundled inside the `.pkl` package for consistent inference at runtime