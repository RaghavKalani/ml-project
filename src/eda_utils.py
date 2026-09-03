from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.graphics.gofplots import qqplot

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "healthcare-dataset-stroke-data.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
NUMERIC_FEATURES = ["age", "bmi", "avg_glucose_level"]
TARGET_COLUMN = "stroke"
CATEGORICAL_FEATURES = [
    "gender",
    "work_type",
    "ever_married",
    "Residence_type",
    "smoking_status",
    "hypertension",
    "heart_disease",
]


def ensure_output_directories() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def find_dataset(root: Path | None = None) -> Path:
    search_root = root or PROJECT_ROOT
    matches = list(search_root.rglob(DATASET_NAME))
    if not matches:
        raise FileNotFoundError(f"Could not locate {DATASET_NAME} under {search_root}")
    return matches[0]


def load_dataset(root: Path | None = None) -> pd.DataFrame:
    dataset_path = find_dataset(root)
    return pd.read_csv(dataset_path)


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def save_figure(filename: str) -> Path:
    ensure_output_directories()
    output_path = FIGURES_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def dataframe_profile(df: pd.DataFrame) -> dict[str, object]:
    categorical_unique = {
        column: df[column].nunique(dropna=False)
        for column in df.select_dtypes(include="object").columns
    }
    missing_values = df.isna().sum()
    missing_percentages = (missing_values / len(df) * 100).round(2)
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes,
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
        "missing_values": missing_values,
        "missing_percentages": missing_percentages,
        "duplicate_rows": int(df.duplicated().sum()),
        "categorical_unique": categorical_unique,
    }


def statistical_summary(df: pd.DataFrame, columns: Iterable[str] = NUMERIC_FEATURES) -> pd.DataFrame:
    summary = pd.DataFrame(index=list(columns))
    for column in columns:
        series = df[column].dropna()
        summary.loc[column, "mean"] = series.mean()
        summary.loc[column, "median"] = series.median()
        summary.loc[column, "mode"] = series.mode().iloc[0]
        summary.loc[column, "variance"] = series.var()
        summary.loc[column, "std_dev"] = series.std()
        summary.loc[column, "skewness"] = series.skew()
        summary.loc[column, "kurtosis"] = series.kurt()
    return summary.round(4)


def normality_summary(df: pd.DataFrame, columns: Iterable[str] = NUMERIC_FEATURES) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in columns:
        series = df[column].dropna()
        statistic, p_value = stats.normaltest(series)
        rows.append(
            {
                "feature": column,
                "normaltest_statistic": statistic,
                "p_value": p_value,
                "normal_at_0.05": bool(p_value >= 0.05),
            }
        )
    return pd.DataFrame(rows)


def encode_for_correlation(df: pd.DataFrame) -> pd.DataFrame:
    encoded = pd.get_dummies(
        df,
        columns=["gender", "ever_married", "work_type", "Residence_type", "smoking_status"],
        drop_first=False,
    )
    return encoded


def outlier_bounds(series: pd.Series) -> tuple[float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)


def outlier_summary(df: pd.DataFrame, columns: Iterable[str] = NUMERIC_FEATURES) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in columns:
        lower_bound, upper_bound = outlier_bounds(df[column].dropna())
        outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        rows.append(
            {
                "feature": column,
                "lower_bound": round(lower_bound, 4),
                "upper_bound": round(upper_bound, 4),
                "outlier_count": int(outlier_mask.sum()),
                "outlier_percentage": round(outlier_mask.mean() * 100, 2),
            }
        )
    return pd.DataFrame(rows)


def target_distribution(df: pd.DataFrame) -> pd.Series:
    return df[TARGET_COLUMN].value_counts().sort_index()


def target_percentages(df: pd.DataFrame) -> pd.Series:
    return (target_distribution(df) / len(df) * 100).round(2)


def create_target_plots(df: pd.DataFrame) -> None:
    set_plot_style()
    counts = target_distribution(df)
    percentages = target_percentages(df)

    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x=TARGET_COLUMN, data=df, palette="Set2")
    ax.set_title("Stroke Target Distribution")
    ax.set_xlabel("Stroke")
    ax.set_ylabel("Count")
    for index, value in enumerate(counts.values):
        ax.text(index, value + 25, f"{value}", ha="center", va="bottom", fontsize=11)
    save_figure("stroke_distribution.png")

    plt.figure(figsize=(7, 7))
    plt.pie(
        counts.values,
        labels=["No Stroke", "Stroke"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#4C78A8", "#F58518"],
        explode=(0, 0.08),
    )
    plt.title("Stroke Class Share")
    plt.tight_layout()
    save_figure("stroke_pie_chart.png")

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(x=percentages.index.astype(str), y=percentages.values, palette="Set1")
    ax.set_title("Stroke Percentage Distribution")
    ax.set_xlabel("Stroke")
    ax.set_ylabel("Percentage")
    for index, value in enumerate(percentages.values):
        ax.text(index, value + 1, f"{value:.2f}%", ha="center", va="bottom", fontsize=11)
    save_figure("stroke_percentage_distribution.png")


def create_numeric_distribution_plots(df: pd.DataFrame, columns: Iterable[str] = NUMERIC_FEATURES) -> None:
    set_plot_style()
    for column in columns:
        plt.figure(figsize=(9, 5))
        ax = sns.histplot(df[column], kde=True, bins=30, color="#4C78A8")
        ax.set_title(f"{column} Distribution")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        save_figure(f"{column}_distribution.png")

        plt.figure(figsize=(9, 5))
        ax = sns.boxplot(x=df[column], color="#F58518")
        ax.set_title(f"{column} Boxplot")
        ax.set_xlabel(column)
        save_figure(f"{column}_boxplot.png")

        plt.figure(figsize=(9, 5))
        ax = sns.violinplot(x=df[column], color="#54A24B", inner="quartile")
        ax.set_title(f"{column} Violin Plot")
        ax.set_xlabel(column)
        save_figure(f"{column}_violinplot.png")


CATEGORY_ORDER = {
    "gender": ["Female", "Male", "Other"],
    "work_type": ["Private", "Self-employed", "children", "Govt_job", "Never_worked"],
    "ever_married": ["Yes", "No"],
    "Residence_type": ["Urban", "Rural"],
    "smoking_status": ["never smoked", "Unknown", "formerly smoked", "smokes"],
    "hypertension": [0, 1],
    "heart_disease": [0, 1],
}


def create_categorical_countplots(df: pd.DataFrame, columns: Iterable[str] = CATEGORICAL_FEATURES) -> None:
    set_plot_style()
    for column in columns:
        plt.figure(figsize=(10, 5))
        order = CATEGORY_ORDER.get(column)
        ax = sns.countplot(x=column, data=df, order=order, palette="viridis")
        ax.set_title(f"{column} Count Plot")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        plt.xticks(rotation=20)
        save_figure(f"{column}_countplot.png")


def create_bivariate_plots(df: pd.DataFrame) -> None:
    set_plot_style()
    plots = [
        ("gender", "count", "stroke_by_gender.png"),
        ("age", "box", "stroke_by_age_boxplot.png"),
        ("bmi", "box", "stroke_by_bmi_boxplot.png"),
        ("avg_glucose_level", "box", "stroke_by_avg_glucose_level_boxplot.png"),
        ("smoking_status", "count", "stroke_by_smoking_status.png"),
        ("work_type", "count", "stroke_by_work_type.png"),
        ("Residence_type", "count", "stroke_by_residence_type.png"),
        ("hypertension", "count", "stroke_by_hypertension.png"),
        ("heart_disease", "count", "stroke_by_heart_disease.png"),
    ]

    for column, plot_kind, filename in plots:
        plt.figure(figsize=(10, 5))
        if plot_kind == "count":
            order = CATEGORY_ORDER.get(column)
            ax = sns.countplot(x=column, hue=TARGET_COLUMN, data=df, order=order, palette="Set2")
            ax.set_ylabel("Count")
        else:
            ax = sns.boxplot(x=TARGET_COLUMN, y=column, data=df, palette="Set3")
            ax.set_ylabel(column)
        ax.set_title(f"Stroke vs {column}")
        ax.set_xlabel(column)
        plt.xticks(rotation=20)
        save_figure(filename)


def create_correlation_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    set_plot_style()
    encoded = encode_for_correlation(df)
    corr = encoded.corr(numeric_only=True)

    plt.figure(figsize=(16, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.2)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    save_figure("correlation_heatmap.png")
    return corr


def create_outlier_plots(df: pd.DataFrame, columns: Iterable[str] = NUMERIC_FEATURES) -> None:
    set_plot_style()
    for column in columns:
        plt.figure(figsize=(9, 5))
        ax = sns.boxplot(x=df[column], color="#9D755D")
        ax.set_title(f"Outlier Check for {column}")
        ax.set_xlabel(column)
        save_figure(f"{column}_outlier_boxplot.png")


def create_distribution_analysis_plots(df: pd.DataFrame, columns: Iterable[str] = NUMERIC_FEATURES) -> None:
    set_plot_style()
    for column in columns:
        plt.figure(figsize=(7, 7))
        qqplot(df[column].dropna(), line="s", ax=plt.gca())
        plt.title(f"QQ Plot - {column}")
        save_figure(f"qq_{column}.png")


def top_correlation_pairs(corr: pd.DataFrame, focus: str = TARGET_COLUMN, top_n: int = 10) -> tuple[pd.Series, pd.Series]:
    focus_corr = corr[focus].sort_values(ascending=False)
    strongest_positive = focus_corr.iloc[1 : top_n + 1]
    strongest_negative = focus_corr.tail(top_n).sort_values()
    return strongest_positive, strongest_negative


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing_values = df.isna().sum()
    return pd.DataFrame(
        {
            "missing_values": missing_values,
            "missing_percentage": (missing_values / len(df) * 100).round(2),
        }
    ).sort_values("missing_values", ascending=False)


def create_missing_values_plot(df: pd.DataFrame) -> None:
    set_plot_style()
    summary = missing_summary(df)
    summary = summary[summary["missing_values"] > 0]
    if summary.empty:
        summary = pd.DataFrame({"missing_percentage": [0]}, index=["No missing values"])

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x=summary.index.astype(str), y=summary["missing_percentage"].values, palette="magma")
    ax.set_title("Missing Value Percentage by Column")
    ax.set_xlabel("Column")
    ax.set_ylabel("Missing Percentage")
    plt.xticks(rotation=30, ha="right")
    save_figure("missing_values_percentage.png")


def dataset_quality_notes(df: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    if df["bmi"].isna().sum() > 0:
        notes.append("BMI contains missing values and should be imputed before model training.")
    if (df["gender"] == "Other").any():
        notes.append("Gender contains a rare 'Other' category with a single record.")
    if (df["work_type"] == "Never_worked").any():
        notes.append("Never_worked is a very small category and may need grouping in modeling.")
    if df.duplicated().sum() == 0:
        notes.append("No duplicate rows were found.")
    return notes


def balanced_label(df: pd.DataFrame) -> bool:
    counts = target_distribution(df)
    minority_share = counts.min() / counts.sum()
    return minority_share >= 0.4
