# Task 4 — Disease Prediction from Medical Data

## Objective
Predict the possibility of diseases based on patient data using classification algorithms.

## Datasets
| Dataset | Source | Features | Target |
|---|---|---|---|
| Heart Disease | UCI ML Repository (synthetic) | Age, Sex, ChestPain, BP, Cholesterol, ECG, MaxHR, etc. | Heart disease (0/1) |
| Diabetes | Pima Indians (synthetic) | Pregnancies, Glucose, BloodPressure, BMI, Insulin, etc. | Diabetic (0/1) |
| Breast Cancer | UCI Breast Cancer Wisconsin (synthetic) | 10 cell nucleus measurements × mean/worst | Malignant/Benign (0/1) |

## Algorithms
- **Logistic Regression** — linear baseline classifier
- **Support Vector Machine (SVM)** — RBF kernel, scaled features
- **Random Forest** — ensemble of 200 decision trees
- **XGBoost** — gradient-boosted trees, 200 estimators

## Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score (test set)
- AUC-ROC (5-fold cross-validation)
- Confusion Matrix (best model per dataset)

## How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Run the prediction pipeline
python DiseasePrediction.py
```

## Outputs
All charts are saved to `outputs/`:
- `heart_disease_results.png` — Heart disease model comparison & analysis
- `diabetes_results.png` — Diabetes model comparison & analysis
- `breast_cancer_results.png` — Breast cancer model comparison & analysis
- `model_comparison_summary.png` — Cross-dataset AUC summary
