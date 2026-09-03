# EDA Report: Stroke Prediction Dataset

## Project Introduction
This project explores the Kaggle stroke prediction dataset to understand data quality, target imbalance, feature distributions, correlations, and outlier behavior before model building.

## Dataset Description
- Shape: 5110 rows and 12 columns
- Memory usage: 1.62 MB
- Target column: stroke
- Numerical features: age, bmi, avg_glucose_level
- Categorical features: gender, work_type, ever_married, Residence_type, smoking_status

## Data Quality Analysis
The dataset is structurally clean, but BMI contains missing values and should be handled before modeling.
- BMI contains missing values and should be imputed before model training.
- Gender contains a rare 'Other' category with a single record.
- Never_worked is a very small category and may need grouping in modeling.
- No duplicate rows were found.

## Missing Values
- Total missing values: 201
- Missing rows are concentrated entirely in bmi.

| Column | Missing Count | Missing % |
| --- | ---: | ---: |
| bmi | 201 | 3.93 |
| id | 0 | 0.00 |
| age | 0 | 0.00 |
| gender | 0 | 0.00 |
| hypertension | 0 | 0.00 |
| heart_disease | 0 | 0.00 |
| work_type | 0 | 0.00 |
| ever_married | 0 | 0.00 |
| Residence_type | 0 | 0.00 |
| avg_glucose_level | 0 | 0.00 |
| smoking_status | 0 | 0.00 |
| stroke | 0 | 0.00 |

## Duplicate Analysis
- Duplicate rows found: 0
- The dataset does not contain exact duplicate records.

## Univariate Analysis
- Stroke class counts: {0: 4861, 1: 249}
- Stroke class percentages: {0: 95.13, 1: 4.87}
- The target distribution is heavily imbalanced, with the non-stroke class dominating the sample.

## Bivariate Analysis
- Higher stroke rates appear among older patients, patients with heart disease, hypertension, and people who are married or self-employed.
- Former smokers show a noticeably higher stroke rate than the other smoking groups.
- Urban versus rural residence shows only a small difference.

## Correlation Analysis
Strongest positive correlations with stroke:
- age: 0.2453
- heart_disease: 0.1349
- avg_glucose_level: 0.1319
- hypertension: 0.1279
- ever_married_Yes: 0.1083
- smoking_status_formerly smoked: 0.0646
- work_type_Self-employed: 0.0622
- bmi: 0.0424
- Residence_type_Urban: 0.0155
- work_type_Private: 0.0119

Strongest negative correlations with stroke:
- ever_married_No: -0.1083
- work_type_children: -0.0839
- smoking_status_Unknown: -0.0559
- Residence_type_Rural: -0.0155
- work_type_Never_worked: -0.0149
- gender_Female: -0.0090
- smoking_status_never smoked: -0.0041
- gender_Other: -0.0032
- work_type_Govt_job: 0.0027
- id: 0.0064

## Outlier Detection
| Feature | Lower Bound | Upper Bound | Outlier Count | Outlier % |
| --- | ---: | ---: | ---: | ---: |
| age | -29.0000 | 115.0000 | 0 | 0.00 |
| bmi | 9.1000 | 47.5000 | 110 | 2.15 |
| avg_glucose_level | 21.9775 | 169.3575 | 627 | 12.27 |

## Statistical Summary
| Feature | Mean | Median | Mode | Variance | Std Dev | Skewness | Kurtosis |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| age | 43.2266 | 45.0000 | 78.0000 | 511.3318 | 22.6126 | -0.1371 | -0.9910 |
| bmi | 28.8932 | 28.1000 | 28.7000 | 61.6864 | 7.8541 | 1.0553 | 3.3627 |
| avg_glucose_level | 106.1477 | 91.8850 | 93.8800 | 2050.6008 | 45.2836 | 1.5723 | 1.6805 |

## Distribution Analysis
| Feature | Normaltest p-value | Normal at 0.05 |
| --- | ---: | --- |
| age | 0.000000 | False |
| bmi | 0.000000 | False |
| avg_glucose_level | 0.000000 | False |

## Key Insights
1. Stroke cases represent a small minority of the dataset, so class imbalance must be handled during modeling.
2. Age is the strongest feature associated with stroke among the encoded variables.
3. Heart disease and hypertension are positively associated with stroke risk.
4. Average glucose level is right-skewed and contains strong high-end outliers.
5. BMI has missing values and moderate outlier behavior.
6. Former smokers and self-employed patients show elevated stroke proportions.
7. Married patients show a higher stroke rate than unmarried patients.
8. Residence type contributes little linear signal in the correlation analysis.
9. The numerical variables are not normally distributed according to the QQ plots and normality tests.
10. The dataset is clean overall, with the main preparation tasks being imputation, encoding, and imbalance handling.

## Conclusion
This EDA shows that the dataset is suitable for classification work, but it is highly imbalanced and requires preprocessing of missing BMI values, categorical encoding, and careful handling of skewed numerical features before training a stroke risk model.
