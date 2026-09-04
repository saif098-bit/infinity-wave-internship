# Heart Disease Prediction

A machine learning project that predicts the presence of heart disease in a patient using the UCI Cleveland Heart Disease dataset. Built as part of the **Infinity Wave Internship**.

## Overview

The notebook walks through a full ML pipeline: data cleaning, exploratory analysis, preprocessing, training/comparing six classification models, hyperparameter tuning, feature selection, and building a leakage-safe final pipeline for deployment-style predictions.

## Dataset

- **Source:** [UCI Heart Disease Data Set](https://www.kaggle.com/datasets/lourenswalters/uci-heart-disease-data-set) (Cleveland subset), loaded via `kagglehub`
- **Samples:** 303 patients, 13 clinical features + target
- **Target:** Presence (1) or absence (0) of heart disease (binarized from the original 0–4 severity scale)

**Features:** age, sex, chest pain type (`cp`), resting blood pressure (`trestbps`), cholesterol (`chol`), fasting blood sugar (`fbs`), resting ECG (`restecg`), max heart rate (`thalach`), exercise-induced angina (`exang`), ST depression (`oldpeak`), slope, number of major vessels (`ca`), and thalassemia (`thal`).

## Approach

1. **Data cleaning** — missing values in `ca` and `thal` imputed with the training-set mode (fit only on train, applied consistently to val/test to avoid leakage).
2. **Split** — 70% train / 15% validation / 15% test, stratified on the target.
3. **Preprocessing** — `StandardScaler` on numerical features, `OneHotEncoder` on categorical features (fit on train only).
4. **Model comparison** (validation set):

   | Model | Accuracy | Precision | Recall | F1 |
   |---|---|---|---|---|
   | Logistic Regression | 82.2% | 84.2% | 76.2% | 80.0% |
   | Decision Tree | 62.2% | 59.1% | 61.9% | 60.5% |
   | Random Forest | 77.8% | 73.9% | 81.0% | 77.3% |
   | SVM | 82.2% | 84.2% | 76.2% | 80.0% |
   | KNN | 82.2% | 88.2% | 71.4% | 78.9% |
   | Tuned Logistic Regression | 75.6% | 77.8% | 66.7% | 71.8% |
   | **Tuned SVM** | **84.4%** | **88.9%** | **76.2%** | **82.1%** |

5. **Hyperparameter tuning** — `GridSearchCV` with 5-fold stratified cross-validation.
6. **Feature selection** — `SelectKBest` (ANOVA F-statistic) tested across feature-count values.
7. **Final pipeline** — a leakage-safe `sklearn.Pipeline` (`ColumnTransformer` + linear-kernel SVM, `C=0.01`) fit on train and evaluated once on the held-out test set.

## Final Results (Test Set)

| Metric | Score |
|---|---|
| Accuracy | **91.30%** |
| Precision | 90.48% |
| Recall | 90.48% |
| F1 Score | 90.48% |

Final model: **Support Vector Machine (linear kernel, C=0.01)** — saved as `final_svm_model.pkl`.

## Project Structure

```
heart-disease-prediction/
├── Heart_Disease_Prediction.ipynb   # Full notebook: EDA → preprocessing → modeling → evaluation
├── final_svm_model.pkl              # Saved final model (generated on run)
├── requirements.txt                 # Python dependencies
└── README.md
```

## How to Run

1. Clone the repo and open the notebook in Jupyter or Google Colab.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run all cells top to bottom. The notebook downloads the dataset automatically via `kagglehub` (requires a free [Kaggle](https://www.kaggle.com) account/API token).
4. The final model is saved as `final_svm_model.pkl` and can be reloaded with `joblib.load()`.

## Making a Prediction

The notebook includes a ready-to-use function:

```python
prediction = predict_heart_disease(
    age=55, sex=1, cp=2, trestbps=130, chol=250,
    fbs=0, restecg=1, thalach=150, exang=0,
    oldpeak=1.0, slope=2, ca=0, thal=3
)
```

## Tech Stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, joblib, kagglehub

## Acknowledgments

Dataset: UCI Machine Learning Repository — Cleveland Heart Disease Data (via Kaggle, hosted by lourenswalters).

