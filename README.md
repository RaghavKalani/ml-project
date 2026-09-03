# Acute Ischemic Stroke Risk Classification Engine - Phase 1 EDA

## Objective
This project performs a complete exploratory data analysis on the Kaggle Stroke Prediction dataset to understand the data quality, class imbalance, feature distributions, correlations, outliers, and statistical properties before building a stroke risk classification model.

## Dataset Source
The dataset used in this project is the Kaggle Stroke Prediction dataset:

- File: `healthcare-dataset-stroke-data.csv`
- Location inside this workspace: `Stroke prediction dataset/healthcare-dataset-stroke-data.csv`

## Folder Structure

```
ML PROJECT
├── Stroke prediction dataset
│   └── healthcare-dataset-stroke-data.csv
├── notebooks
│   ├── 01_EDA.ipynb
│   └── 02_modeling.ipynb
├── models
│   └── stroke_model.pkl
├── reports
│   └── EDA_Report.md
├── figures
├── src
├── requirements.txt
├── README.md
└── START.md
```

## Installation
1. Create and activate a virtual environment.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Required Libraries
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- scipy
- statsmodels
- jupyter
- notebook
- ipykernel
- python-docx
- reportlab

## How to Run
1. Open the project folder in VS Code.
2. Open `notebooks/01_EDA.ipynb`.
3. Select the Python interpreter from your virtual environment.
4. Run the notebook cells from top to bottom.
5. Review the generated figures in `figures/` and the report in `reports/EDA_Report.md`.

You can also regenerate the analysis artifacts from the command line:

```bash
python -m src.generate_eda
```

## Phase 2: Model Training and Testing

After installing the dependencies and activating `.venv`, run:

```bash
python -m src.train_model
```

This trains Logistic Regression and Random Forest pipelines, handles missing values and categorical encoding without data leakage, evaluates precision, recall, F1-score, ROC-AUC, and PR-AUC, tests several classification thresholds, and performs 5-fold stratified cross-validation. The selected pipeline is saved to `models/stroke_model.pkl`.

You can also open `notebooks/02_modeling.ipynb` and run its single code cell. Model results are saved in `reports/ML_Model_Report.md`, `reports/model_metrics.csv`, `reports/threshold_metrics.csv`, and `reports/cross_validation_pr_auc.txt`.

## Results
The current EDA shows that:
- The dataset contains 5,110 rows and 12 columns.
- The target variable is highly imbalanced.
- BMI contains missing values.
- Age, heart disease, hypertension, and glucose level are the strongest signals associated with stroke.
- Numerical variables are not normally distributed.

## Future Work
- Impute missing BMI values.
- Encode categorical variables for modeling.
- Handle class imbalance with resampling or class weights.
- Train and evaluate baseline and advanced classification models.
- Compare model performance using recall, precision, F1-score, and ROC-AUC.
