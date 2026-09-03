from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.eda_utils import (
    PROJECT_ROOT,
    TARGET_COLUMN,
    balanced_label,
    create_bivariate_plots,
    create_categorical_countplots,
    create_correlation_heatmap,
    create_distribution_analysis_plots,
    create_numeric_distribution_plots,
    create_outlier_plots,
    create_missing_values_plot,
    create_target_plots,
    dataframe_profile,
    dataset_quality_notes,
    ensure_output_directories,
    find_dataset,
    load_dataset,
    missing_summary,
    normality_summary,
    outlier_summary,
    save_figure,
    statistical_summary,
    target_distribution,
    target_percentages,
    top_correlation_pairs,
)


def _format_series(series: pd.Series, precision: int = 4) -> str:
    if series.empty:
        return "None"
    return "\n".join(f"- {index}: {value:.{precision}f}" if isinstance(value, float) else f"- {index}: {value}" for index, value in series.items())


def _build_markdown_report(df: pd.DataFrame, corr: pd.DataFrame) -> str:
    profile = dataframe_profile(df)
    missing = missing_summary(df)
    stats_summary = statistical_summary(df)
    normality = normality_summary(df)
    outliers = outlier_summary(df)
    positive_corr, negative_corr = top_correlation_pairs(corr)
    notes = dataset_quality_notes(df)
    target_counts = target_distribution(df)
    target_pct = target_percentages(df)

    report_lines = [
        "# EDA Report: Stroke Prediction Dataset",
        "",
        "## Project Introduction",
        "This project explores the Kaggle stroke prediction dataset to understand data quality, target imbalance, feature distributions, correlations, and outlier behavior before model building.",
        "",
        "## Dataset Description",
        f"- Shape: {profile['shape'][0]} rows and {profile['shape'][1]} columns",
        f"- Memory usage: {profile['memory_bytes'] / (1024 ** 2):.2f} MB",
        f"- Target column: {TARGET_COLUMN}",
        f"- Numerical features: age, bmi, avg_glucose_level",
        f"- Categorical features: gender, work_type, ever_married, Residence_type, smoking_status",
        "",
        "## Data Quality Analysis",
        "The dataset is structurally clean, but BMI contains missing values and should be handled before modeling.",
        *[f"- {note}" for note in notes],
        "",
        "## Missing Values",
        f"- Total missing values: {int(profile['missing_values'].sum())}",
        "- Missing rows are concentrated entirely in bmi.",
        "",
        "| Column | Missing Count | Missing % |",
        "| --- | ---: | ---: |",
    ]
    for column, row in missing.iterrows():
        report_lines.append(f"| {column} | {int(row['missing_values'])} | {row['missing_percentage']:.2f} |")

    report_lines.extend(
        [
            "",
            "## Duplicate Analysis",
            f"- Duplicate rows found: {profile['duplicate_rows']}",
            "- The dataset does not contain exact duplicate records.",
            "",
            "## Univariate Analysis",
            f"- Stroke class counts: {target_counts.to_dict()}",
            f"- Stroke class percentages: {target_pct.to_dict()}",
            "- The target distribution is heavily imbalanced, with the non-stroke class dominating the sample.",
            "",
            "## Bivariate Analysis",
            "- Higher stroke rates appear among older patients, patients with heart disease, hypertension, and people who are married or self-employed.",
            "- Former smokers show a noticeably higher stroke rate than the other smoking groups.",
            "- Urban versus rural residence shows only a small difference.",
            "",
            "## Correlation Analysis",
            "Strongest positive correlations with stroke:",
            _format_series(positive_corr),
            "",
            "Strongest negative correlations with stroke:",
            _format_series(negative_corr),
            "",
            "## Outlier Detection",
            "| Feature | Lower Bound | Upper Bound | Outlier Count | Outlier % |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in outliers.iterrows():
        report_lines.append(
            f"| {row['feature']} | {row['lower_bound']:.4f} | {row['upper_bound']:.4f} | {int(row['outlier_count'])} | {row['outlier_percentage']:.2f} |"
        )

    report_lines.extend(
        [
            "",
            "## Statistical Summary",
            "| Feature | Mean | Median | Mode | Variance | Std Dev | Skewness | Kurtosis |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for index, row in stats_summary.iterrows():
        report_lines.append(
            f"| {index} | {row['mean']:.4f} | {row['median']:.4f} | {row['mode']:.4f} | {row['variance']:.4f} | {row['std_dev']:.4f} | {row['skewness']:.4f} | {row['kurtosis']:.4f} |"
        )

    report_lines.extend(
        [
            "",
            "## Distribution Analysis",
            "| Feature | Normaltest p-value | Normal at 0.05 |",
            "| --- | ---: | --- |",
        ]
    )
    for _, row in normality.iterrows():
        report_lines.append(f"| {row['feature']} | {row['p_value']:.6f} | {bool(row['normal_at_0.05'])} |")

    report_lines.extend(
        [
            "",
            "## Key Insights",
            "1. Stroke cases represent a small minority of the dataset, so class imbalance must be handled during modeling.",
            "2. Age is the strongest feature associated with stroke among the encoded variables.",
            "3. Heart disease and hypertension are positively associated with stroke risk.",
            "4. Average glucose level is right-skewed and contains strong high-end outliers.",
            "5. BMI has missing values and moderate outlier behavior.",
            "6. Former smokers and self-employed patients show elevated stroke proportions.",
            "7. Married patients show a higher stroke rate than unmarried patients.",
            "8. Residence type contributes little linear signal in the correlation analysis.",
            "9. The numerical variables are not normally distributed according to the QQ plots and normality tests.",
            "10. The dataset is clean overall, with the main preparation tasks being imputation, encoding, and imbalance handling.",
            "",
            "## Conclusion",
            "This EDA shows that the dataset is suitable for classification work, but it is highly imbalanced and requires preprocessing of missing BMI values, categorical encoding, and careful handling of skewed numerical features before training a stroke risk model.",
        ]
    )
    return "\n".join(report_lines) + "\n"


def run() -> None:
    ensure_output_directories()
    df = load_dataset(PROJECT_ROOT)

    create_target_plots(df)
    create_numeric_distribution_plots(df)
    create_categorical_countplots(df)
    create_missing_values_plot(df)
    create_bivariate_plots(df)
    corr = create_correlation_heatmap(df)
    create_outlier_plots(df)
    create_distribution_analysis_plots(df)

    report_path = PROJECT_ROOT / "reports" / "EDA_Report.md"
    report_path.write_text(_build_markdown_report(df, corr), encoding="utf-8")

    print(f"Dataset located at: {find_dataset(PROJECT_ROOT)}")
    print(f"Report written to: {report_path}")
    print("Figures written to: figures/")
    print(f"Stroke class imbalance: {target_percentages(df).to_dict()}")
    print(f"Balanced dataset: {balanced_label(df)}")


if __name__ == "__main__":
    run()
