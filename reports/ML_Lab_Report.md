# Machine Learning Laboratory

Experiment:
Acute Ischemic Stroke Risk Classification Engine – Dataset Exploration & Exploratory Data Analysis

Dataset:
Healthcare Stroke Prediction Dataset

--------------------------------------------------

Question 1

Aim
Import, load and view the dataset.

Program
```python
from pathlib import Path
import sys

import pandas as pd
from IPython.display import display

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / 'Stroke prediction dataset').exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eda_utils import (
    create_bivariate_plots,
    create_categorical_countplots,
    create_correlation_heatmap,
    create_distribution_analysis_plots,
    create_missing_values_plot,
    create_numeric_distribution_plots,
    create_outlier_plots,
    create_target_plots,
    dataframe_profile,
    dataset_quality_notes,
    ensure_output_directories,
    load_dataset,
    missing_summary,
    normality_summary,
    outlier_summary,
    statistical_summary,
    target_distribution,
    target_percentages,
)

ensure_output_directories()
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)

df = load_dataset(PROJECT_ROOT)

print('First 5 rows:')
display(df.head())

print('Last 5 rows:')
display(df.tail())

print('Dataset shape:', df.shape)
print('Column names:', list(df.columns))
print('Data types:')
display(df.dtypes.to_frame(name='dtype'))
print(f'Memory usage: {df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB')
```

Output
![](figures/q1_head_output.png)

Figure 1. First five rows of the dataset.

![](figures/q1_tail_output.png)

Figure 2. Last five rows of the dataset.

![](figures/q1_shape_columns_output.png)

Figure 3. Dataset shape and column names.

![](figures/q1_dtypes_output.png)

Figure 4. Data types for each column.

![](figures/q1_info_output.png)

Figure 5. `info()` output for the dataset.

Analysis
- The dataset loads correctly with 5110 rows and 12 columns.
- The file includes the expected stroke-related variables and mixed data types.
- The shape, column names, and memory usage confirm that the data is small enough for manual EDA.
- No loading issues appear in the notebook outputs.

Conclusion
The dataset loads successfully and is ready for EDA.

--------------------------------------------------

Question 2

Aim
Display dataset overview and summary statistics.

Program
```python
df.info()

print('\nNumerical summary:')
display(df.describe())

print('Categorical summary:')
display(df.describe(include='object'))
```

Output
![](figures/q2_describe_numeric_output.png)

Figure 6. Summary statistics for numerical columns.

![](figures/q2_describe_object_output.png)

Figure 7. Summary statistics for categorical columns.

![](figures/q2_missing_values_output.png)

Figure 8. Missing value summary table.

![](figures/q2_duplicate_count_output.png)

Figure 9. Duplicate row count.

Analysis
- The numerical summary shows that age ranges from 0.08 to 82.00.
- Average glucose level is strongly right-skewed, while BMI is moderately right-skewed.
- `describe(include='object')` confirms that Female, Private, Yes, Urban, and never smoked are the dominant categories.
- The data types remain consistent with a structured tabular dataset.

Conclusion
The summary statistics show a realistic clinical dataset with skewness and missing BMI values.

--------------------------------------------------

Question 3

Aim
Analyze data quality.

Program
```python
profile = dataframe_profile(df)
missing_df = missing_summary(df)

display(missing_df)
print('Total missing values:', int(profile['missing_values'].sum()))
print('Duplicate rows:', profile['duplicate_rows'])

categorical_columns = df.select_dtypes(include='object').columns.tolist()
for column in categorical_columns:
    print(f'\nUnique values for {column}:')
    display(df[column].value_counts(dropna=False).to_frame(name='count'))

print('Quality notes:')
for note in dataset_quality_notes(df):
    print('-', note)

create_missing_values_plot(df)
```

Output
![](figures/q3_missing_table.png)

Figure 10. Missing value table.

![](figures/missing_values_heatmap.png)

Figure 11. Missing value heatmap.

![](figures/q3_unique_gender.png)

Figure 12. Unique categorical values for gender.

![](figures/q3_unique_work_type.png)

Figure 13. Unique categorical values for work type.

![](figures/q3_unique_ever_married.png)

Figure 14. Unique categorical values for ever married.

![](figures/q3_unique_Residence_type.png)

Figure 15. Unique categorical values for Residence type.

![](figures/q3_unique_smoking_status.png)

Figure 16. Unique categorical values for smoking status.

Analysis
- BMI is the only column with missing values, and it contributes 201 missing records.
- The heatmap shows a very narrow missingness pattern rather than scattered nulls.
- Duplicate rows are absent, which simplifies downstream preprocessing.
- The categorical variables have a few rare levels such as Other and Never_worked.

Conclusion
The data quality is good overall, with BMI missingness as the main issue.

--------------------------------------------------

Question 4

Aim
Analyze the target variable.

Program
```python
target_df = pd.DataFrame({
    'count': target_distribution(df),
    'percentage': target_percentages(df),
})
display(target_df)
create_target_plots(df)
```

Output
![](figures/stroke_distribution.png)

Figure 17. Stroke count plot.

![](figures/stroke_pie_chart.png)

Figure 18. Stroke pie chart.

![](figures/stroke_percentage_distribution.png)

Table 1. Stroke class percentage distribution.

| Stroke | Count | Percentage |
| --- | --- | --- |
| 0 | 4861 | 95.13% |
| 1 | 249 | 4.87% |

Analysis
- Stroke class 0 contains 4861 records (95.13%).
- Stroke class 1 contains 249 records (4.87%).
- The class distribution is highly imbalanced and accuracy alone would be misleading.
- The pie chart and count plot both show that stroke is a minority class.

Conclusion
Stroke is a severely imbalanced target variable.

--------------------------------------------------

Question 5

Aim
Analyze numerical features.

Program
```python
display(statistical_summary(df))
create_numeric_distribution_plots(df)
```

Output
![](figures/age_distribution.png)

Figure 20. Age histogram.

![](figures/bmi_distribution.png)

Figure 21. BMI histogram.

![](figures/avg_glucose_level_distribution.png)

Figure 22. Glucose histogram.

![](figures/age_boxplot.png)

Figure 23. Age boxplot.

![](figures/bmi_boxplot.png)

Figure 24. BMI boxplot.

![](figures/avg_glucose_level_boxplot.png)

Figure 25. Glucose boxplot.

![](figures/age_violinplot.png)

Figure 26. Age violin plot.

![](figures/bmi_violinplot.png)

Figure 27. BMI violin plot.

![](figures/avg_glucose_level_violinplot.png)

Figure 28. Glucose violin plot.

Analysis
- Age is broadly spread and visually close to a symmetric distribution.
- BMI is right-skewed with a long upper tail.
- Average glucose level has the strongest skewness and the most extreme values.
- The violin plots confirm that the central mass of BMI and glucose is concentrated below the upper tails.

Conclusion
The numerical variables are skewed and require preprocessing.

--------------------------------------------------

Question 6

Aim
Analyze categorical variables.

Program
```python
create_categorical_countplots(df)
for column in ['gender', 'work_type', 'ever_married', 'Residence_type', 'smoking_status', 'hypertension', 'heart_disease']:
    print(f'\n{column}:')
    display(df[column].value_counts(dropna=False).to_frame(name='count'))
```

Output
![](figures/gender_countplot.png)

Figure 29. Gender count plot.

![](figures/work_type_countplot.png)

Figure 30. Work type count plot.

![](figures/ever_married_countplot.png)

Figure 31. Ever married count plot.

![](figures/Residence_type_countplot.png)

Figure 32. Residence type count plot.

![](figures/smoking_status_countplot.png)

Figure 33. Smoking status count plot.

![](figures/hypertension_countplot.png)

Figure 34. Hypertension count plot.

![](figures/heart_disease_countplot.png)

Figure 35. Heart disease count plot.

Analysis
- Female records are more common than male records, with a single Other record.
- Private work type is the largest category, followed by Self-employed and children.
- Residence type is nearly balanced between Urban and Rural.
- Smoking status includes a large Unknown group, which should be handled carefully later.

Conclusion
The categorical variables are clean but contain a few rare levels.

--------------------------------------------------

Question 7

Aim
Perform bivariate analysis.

Program
```python
create_bivariate_plots(df)
```

Output
![](figures/stroke_by_gender.png)

Figure 36. Stroke by gender.

![](figures/stroke_by_age_boxplot.png)

Figure 37. Stroke by age.

![](figures/stroke_by_bmi_boxplot.png)

Figure 38. Stroke by BMI.

![](figures/stroke_by_avg_glucose_level_boxplot.png)

Figure 39. Stroke by glucose level.

![](figures/stroke_by_smoking_status.png)

Figure 40. Stroke by smoking status.

![](figures/stroke_by_work_type.png)

Figure 41. Stroke by work type.

![](figures/stroke_by_residence_type.png)

Figure 42. Stroke by residence type.

![](figures/stroke_by_hypertension.png)

Figure 43. Stroke by hypertension.

![](figures/stroke_by_heart_disease.png)

Figure 44. Stroke by heart disease.

Analysis
- Stroke is visibly more common among older patients.
- Patients with hypertension or heart disease show a higher stroke proportion.
- Former smokers and self-employed patients show elevated stroke proportions.
- Residence type has a much weaker relationship with stroke than the clinical factors.

Conclusion
Older age, hypertension, heart disease, and glucose are the clearest bivariate signals.

--------------------------------------------------

Question 8

Aim
Perform correlation analysis.

Program
```python
corr = create_correlation_heatmap(df)
display(corr['stroke'].sort_values(ascending=False).to_frame(name='correlation_to_stroke'))
```

Output
![](figures/correlation_matrix.png)

Figure 45. Correlation matrix.

![](figures/correlation_heatmap.png)

Figure 46. Correlation heatmap.

| Feature | Correlation with Stroke |
| --- | --- |
| stroke | 1.000 |
| age | 0.245 |
| heart_disease | 0.135 |
| avg_glucose_level | 0.132 |
| hypertension | 0.128 |
| ever_married_Yes | 0.108 |
| smoking_status_formerly smoked | 0.065 |
| work_type_Self-employed | 0.062 |
| bmi | 0.042 |
| Residence_type_Urban | 0.015 |
| work_type_Private | 0.012 |
| gender_Male | 0.009 |
| smoking_status_smokes | 0.009 |
| id | 0.006 |
| work_type_Govt_job | 0.003 |
| gender_Other | -0.003 |
| smoking_status_never smoked | -0.004 |
| gender_Female | -0.009 |
| work_type_Never_worked | -0.015 |
| Residence_type_Rural | -0.015 |
| smoking_status_Unknown | -0.056 |
| work_type_children | -0.084 |
| ever_married_No | -0.108 |

Analysis
- Age has the strongest positive correlation with stroke among the encoded variables.
- Heart disease, average glucose level, hypertension, and marital status also show positive association.
- Being unmarried and being in the children work type are negatively correlated with stroke.
- The heatmap shows moderate, not extreme, relationships overall.

Conclusion
Age is the strongest positive correlation with stroke.

--------------------------------------------------

Question 9

Aim
Perform outlier detection and distribution analysis.

Program
```python
display(outlier_summary(df))
create_outlier_plots(df)

display(normality_summary(df))
create_distribution_analysis_plots(df)
```

Output
![](figures/age_outlier_boxplot.png)

Figure 47. Age outlier boxplot.

![](figures/bmi_outlier_boxplot.png)

Figure 48. BMI outlier boxplot.

![](figures/avg_glucose_level_outlier_boxplot.png)

Figure 49. Glucose outlier boxplot.

![](figures/qq_age.png)

Figure 50. Age QQ plot.

![](figures/qq_bmi.png)

Figure 51. BMI QQ plot.

![](figures/qq_avg_glucose_level.png)

Figure 52. Glucose QQ plot.

| Feature | Lower Bound | Upper Bound | Outlier Count | Outlier % |
| --- | --- | --- | --- | --- |
| age | -29.0000 | 115.0000 | 0 | 0.00% |
| bmi | 9.1000 | 47.5000 | 110 | 2.15% |
| avg_glucose_level | 21.9775 | 169.3575 | 627 | 12.27% |

| Feature | Normaltest Statistic | P-value | Normal at 0.05 |
| --- | --- | --- | --- |
| age | 1120.5286 | 0.000000 | False |
| bmi | 1021.1795 | 0.000000 | False |
| avg_glucose_level | 1328.9358 | 0.000000 | False |

Analysis
- Outlier counts under the IQR rule are 0 for age, 110 for BMI, and 627 for glucose.
- Average glucose level has the clearest outlier spread.
- The QQ plots confirm that all three numerical variables deviate from normality.
- Normality testing marks age as False, BMI as False, and glucose as False at the 0.05 level.

Conclusion
The numerical features are non-normal and outlier-prone.

--------------------------------------------------

Analysis
- Dataset dimensions: 5,110 rows and 12 columns.
- Missing values: 201 missing BMI values.
- Duplicate rows: 0.
- Numerical features: age, bmi, avg_glucose_level.
- Categorical features: gender, work_type, ever_married, Residence_type, smoking_status.
- Target imbalance: stroke class 0 dominates at 95.13% while class 1 is only 4.87%.
- Age has the strongest relationship with stroke.
- Glucose and heart disease also show clear positive association with stroke.
- BMI is missing for a subset of rows and is moderately skewed.
- Hypertension and heart disease are important clinical signals.
- Outliers are most visible in glucose, followed by BMI.
- Correlations are moderate rather than extreme, which is useful for classification.
- The numerical variables are not normally distributed.
- Preprocessing should include imputation, encoding, and imbalance handling.

Conclusion
The stroke prediction dataset is suitable for machine learning, but preprocessing is required before modeling. The EDA shows a strongly imbalanced target, missing BMI values, right-skewed numerical variables, and clinically meaningful relationships with age, glucose level, hypertension, and heart disease. These patterns suggest that the dataset can support a classification model, but only after imputation, encoding, scaling, and imbalance-aware training are applied. The current dataset quality is good enough to proceed to the next phase, provided that the preprocessing pipeline is carefully designed.

Additional Figures

Figure A1. Missing Values Percentage.
![](figures/missing_values_percentage.png)
