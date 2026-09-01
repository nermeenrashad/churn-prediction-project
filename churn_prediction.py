
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATA_PATH = "ecommerce_customer_churn_dataset.csv"
OUTPUT_DIR = "outputs"
RANDOM_STATE = 42
TEST_SIZE = 0.2

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    """Load the raw dataset from disk."""
    df = pd.read_csv(path)
    return df


def detect_target_column(df: pd.DataFrame) -> str:
    """Automatically detect the churn/target column by name."""
    candidates = [c for c in df.columns if "churn" in c.lower()]
    if not candidates:
        raise ValueError(
            "Could not automatically detect a churn/target column. "
            "Please set TARGET_COLUMN manually."
        )
    return candidates[0]


def clean_data(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Handle missing values, encode target, and drop identifier columns."""
    df = df.copy()

    # Normalize target column to binary 0/1
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].map(
            {"Yes": 1, "No": 0, "yes": 1, "no": 0, "True": 1, "False": 0}
        ).fillna(df[target_col])
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    df = df.dropna(subset=[target_col])
    df[target_col] = df[target_col].astype(int)

    # Fill missing numeric values with the median
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill missing categorical values with the mode
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Drop obvious identifier columns (high-cardinality, not predictive)
    id_like_cols = [
        c for c in df.columns
        if c != target_col and df[c].nunique() > 0.9 * len(df)
        and ("id" in c.lower() or "customer" in c.lower())
    ]
    df = df.drop(columns=id_like_cols, errors="ignore")

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all remaining categorical columns."""
    df = df.copy()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df


def tune_decision_tree(X_train, y_train) -> DecisionTreeClassifier:
    """Search for better Decision Tree hyperparameters via cross-validation.

    A default Decision Tree tends to underperform other ensembles; tuning
    depth and leaf size against ROC-AUC noticeably closes that gap.
    """
    param_grid = {
        "max_depth": [4, 6, 8, 10, 12, None],
        "min_samples_leaf": [1, 5, 10, 20],
        "criterion": ["gini", "entropy"],
    }
    base_tree = DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced")
    search = GridSearchCV(
        base_tree, param_grid, scoring="roc_auc", cv=5, n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


def get_models(tuned_tree: DecisionTreeClassifier) -> dict:
    """Define the four models to train and compare."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "Decision Tree": tuned_tree,
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
        "SVM (RBF Kernel)": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


# Models that require feature scaling (distance/margin-based)
SCALED_MODELS = {"SVM (RBF Kernel)"}


def evaluate_model(y_true, y_pred, y_proba) -> dict:
    """Compute standard classification metrics."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
    }


def plot_model_comparison(results_df: pd.DataFrame, output_path: str):
    """Bar chart comparing all metrics across models."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    plot_df = results_df.set_index("Model")[metrics]

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df.plot(kind="bar", ax=ax)
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrices(predictions: dict, y_test, output_path: str):
    """Confusion matrix grid, one panel per model."""
    fig, axes = plt.subplots(1, len(predictions), figsize=(6 * len(predictions), 5))
    if len(predictions) == 1:
        axes = [axes]

    for ax, (name, (y_pred, _)) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
        )
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names, output_path: str, top_n: int = 15):
    """Horizontal bar chart of the top feature importances."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    importances.plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title("Top Features Driving Churn (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return importances


def plot_shap_summary(model, X_test, output_dir: str):
    """Generate SHAP summary plots for model interpretability."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    # 1. Load and prepare data
    df = load_data(DATA_PATH)
    target_col = detect_target_column(df)
    df = clean_data(df, target_col)
    df = encode_categoricals(df)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # SVM needs scaled features; tree-based models do not
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. Tune the Decision Tree, then train and evaluate all models
    tuned_tree = tune_decision_tree(X_train, y_train)
    models = get_models(tuned_tree)
    results = []
    trained_models = {}
    predictions = {}

    for name, model in models.items():
        if name in SCALED_MODELS:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        elif name == "Decision Tree":
            # already fitted by GridSearchCV during tuning
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

        trained_models[name] = model
        predictions[name] = (y_pred, y_proba)

        metrics = evaluate_model(y_test, y_pred, y_proba)
        metrics["Model"] = name
        results.append(metrics)

    results_df = pd.DataFrame(results)[
        ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    ].sort_values("ROC-AUC", ascending=False).reset_index(drop=True)

    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_results.csv"), index=False)

    # 3. Visualizations
    plot_model_comparison(results_df, os.path.join(OUTPUT_DIR, "model_comparison.png"))
    plot_confusion_matrices(predictions, y_test, os.path.join(OUTPUT_DIR, "confusion_matrices.png"))

    rf_model = trained_models["Random Forest"]
    importances = plot_feature_importance(
        rf_model, X.columns, os.path.join(OUTPUT_DIR, "feature_importance.png")
    )

    # 4. Model interpretability (SHAP) — must use a tree-based model
    tree_based_models = {"Random Forest", "Decision Tree", "Gradient Boosting"}
    best_model_name = results_df.iloc[0]["Model"]
    shap_model_name = best_model_name if best_model_name in tree_based_models else "Random Forest"
    shap_model = trained_models[shap_model_name]
    plot_shap_summary(shap_model, X_test, OUTPUT_DIR)

    # 5. Classification reports
    report_lines = []
    for name, (y_pred, _) in predictions.items():
        report_lines.append(f"{name}\n{'-' * len(name)}")
        report_lines.append(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    with open(os.path.join(OUTPUT_DIR, "classification_reports.txt"), "w") as f:
        f.write("\n".join(report_lines))

    print(f"Best model: {best_model_name}")
    print(results_df.to_string(index=False))
    print(f"All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
