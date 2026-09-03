from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.eda_utils import (
    CATEGORICAL_FEATURES,
    FIGURES_DIR,
    NUMERIC_FEATURES,
    PROJECT_ROOT,
    TARGET_COLUMN,
    ensure_output_directories,
    load_dataset,
)

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "model_metrics.csv"
REPORT_PATH = REPORTS_DIR / "ML_Model_Report.md"
RANDOM_STATE = 42


def make_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def make_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        min_samples_leaf=3,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def evaluate_thresholds(y_true: pd.Series, probabilities: pd.Series) -> pd.DataFrame:
    rows = []
    for threshold in [0.20, 0.30, 0.40, 0.50, 0.60]:
        predictions = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_true, predictions, zero_division=0),
                "recall": recall_score(y_true, predictions, zero_division=0),
                "f1": f1_score(y_true, predictions, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def save_confusion_matrix(name: str, y_true: pd.Series, predictions: pd.Series) -> None:
    display = ConfusionMatrixDisplay(confusion_matrix(y_true, predictions))
    display.plot(cmap="Blues", values_format="d")
    plt.title(f"{name.replace('_', ' ').title()} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{name}_confusion_matrix.png", dpi=300)
    plt.close()


def build_report(metrics: pd.DataFrame, threshold_table: pd.DataFrame, best_name: str) -> str:
    best = metrics.loc[metrics["model"] == best_name].iloc[0]
    lines = [
        "# ML Model Report: Stroke Prediction",
        "",
        "## Method",
        "- The target is `stroke`; the identifier column was excluded.",
        "- Missing BMI values are median-imputed inside the pipeline.",
        "- Categorical variables are one-hot encoded inside the pipeline.",
        "- The data uses an 80/20 stratified train/test split with `random_state=42`.",
        "- Logistic Regression and Random Forest use balanced class weights because stroke is rare.",
        "",
        "## Test-set Results",
        metrics.to_markdown(index=False),
        "",
        f"The selected baseline is **{best_name.replace('_', ' ').title()}**, based on the highest test PR-AUC ({best['pr_auc']:.4f}).",
        "",
        "## Threshold Analysis",
        threshold_table.to_markdown(index=False),
        "",
        "A threshold below 0.50 can improve recall, but it also increases false positives. The final threshold should be chosen according to the application's clinical or operational cost of missed cases versus unnecessary alerts.",
        "",
        "## Validation Notes",
        "- PR-AUC and recall are emphasized because the target is highly imbalanced.",
        "- These results are an educational modeling baseline, not a clinical diagnostic system.",
    ]
    return "\n".join(lines) + "\n"


def run() -> None:
    ensure_output_directories()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset(PROJECT_ROOT)
    X = df.drop(columns=[TARGET_COLUMN, "id"])
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    rows = []
    fitted_models: dict[str, Pipeline] = {}
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        rows.append(
            {
                "model": name,
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1": f1_score(y_test, predictions, zero_division=0),
                "roc_auc": roc_auc_score(y_test, probabilities),
                "pr_auc": average_precision_score(y_test, probabilities),
            }
        )
        fitted_models[name] = model
        save_confusion_matrix(name, y_test, predictions)
        print(f"{name}:\n{classification_report(y_test, predictions, zero_division=0)}")

    metrics = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    best_name = str(metrics.iloc[0]["model"])
    best_model = fitted_models[best_name]
    best_probabilities = best_model.predict_proba(X_test)[:, 1]
    thresholds = evaluate_thresholds(y_test, pd.Series(best_probabilities))
    metrics.to_csv(METRICS_PATH, index=False)
    thresholds.to_csv(REPORTS_DIR / "threshold_metrics.csv", index=False)
    joblib.dump(best_model, MODELS_DIR / "stroke_model.pkl")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(make_models()[best_name], X, y, cv=cv, scoring="average_precision")
    (REPORTS_DIR / "cross_validation_pr_auc.txt").write_text(
        "5-fold stratified PR-AUC scores: " + ", ".join(f"{score:.4f}" for score in cv_scores) + "\n"
        f"Mean PR-AUC: {cv_scores.mean():.4f}\n",
        encoding="utf-8",
    )
    REPORT_PATH.write_text(build_report(metrics, thresholds, best_name), encoding="utf-8")
    print(f"Best model: {best_name}")
    print(f"Mean 5-fold PR-AUC: {cv_scores.mean():.4f}")
    print(f"Saved model: {MODELS_DIR / 'stroke_model.pkl'}")


if __name__ == "__main__":
    run()
