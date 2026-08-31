# Customer Churn Prediction

A machine learning project that predicts customer churn in an e-commerce
business using tabular customer data, and compares the performance of
three classification models.

## Problem

Customer churn (losing customers) is one of the most expensive problems
for subscription and e-commerce businesses. Identifying customers who are
likely to churn — and understanding *why* — lets a business intervene
before it's too late.

## Dataset

`ecommerce_customer_churn_dataset.csv` — customer-level features
(e.g. tenure, order frequency, spend, support tickets, satisfaction score)
with a binary churn label.

## Models Compared

| Model | Why it was chosen |
|---|---|
| **Random Forest** | Ensemble of decision trees; strong baseline for tabular data, resistant to overfitting. |
| **Decision Tree (tuned)** | Simple, fully interpretable. Hyperparameters (`max_depth`, `min_samples_leaf`, `criterion`) are automatically tuned with 5-fold cross-validation (`GridSearchCV`, optimizing ROC-AUC) instead of using arbitrary defaults, so it competes fairly with the ensembles. |
| **Gradient Boosting** | Sequential boosting ensemble — builds trees that correct the errors of previous ones. Typically one of the strongest performers on tabular data like this. |
| **SVM (RBF Kernel)** | Margin-based classifier, good at capturing non-linear decision boundaries; a fundamentally different approach from the tree-based models, useful for comparison. Requires feature scaling. |

Each model is evaluated on: **Accuracy, Precision, Recall, F1-Score, and
ROC-AUC**, since churn datasets are typically imbalanced and accuracy
alone can be misleading.

## Interpretability

- **Feature Importance** (Random Forest) — which features matter most overall.
- **SHAP values** — explains individual predictions and shows how each
  feature pushes a customer toward or away from churning, computed on the
  best-performing model.

## Project Structure

```
customer_churn_project/
├── data/
│   └── ecommerce_customer_churn_dataset.csv   # place the dataset here
├── src/
│   └── churn_prediction.py                    # main script
├── outputs/                                    # generated after running
│   ├── model_comparison_results.csv
│   ├── model_comparison.png
│   ├── confusion_matrices.png
│   ├── feature_importance.png
│   ├── shap_summary.png
│   ├── shap_feature_importance.png
│   └── classification_reports.txt
├── requirements.txt
└── README.md
```

## How to Run

1. Place the dataset inside the `data/` folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script from inside `src/`:
   ```bash
   cd src
   python churn_prediction.py
   ```
4. Results (plots + CSV + reports) will be saved automatically to `outputs/`.

## Results

After running, see `outputs/model_comparison_results.csv` for the full
metrics table and `outputs/model_comparison.png` for a visual comparison
of the three models.
