# Bank Marketing Dataset - MLflow Project
    1-Table of Contents
    2-Introduction
    3- Dataset Description
    4- Features
    5- Setup and Installation
    6- Project Workflow
    7- MLflow Integration
    8- Results and Evaluation
    9- Contributing
    10- License
# Introduction
The Bank Marketing Dataset MLflow Project is a machine learning project that predicts whether a client will subscribe to a term deposit (deposit as target variable) based on their demographic and interaction data. This repository utilizes MLflow to streamline experiment tracking, reproducibility, and model deployment.

# Dataset Description
The dataset used in this project is from the Bank Marketing Dataset on Kaggle.

- Source: UCI Machine Learning Repository
- Size: ~45,000 rows and 17 features
- Objective: Predict the outcome of the marketing campaign (deposit: yes/no).
# Features
Key features include:

- Demographic Information: age, job, marital, education.
- Campaign Details: campaign, pdays, previous, poutcome.
- Financial Data: balance, loan, housing.
- Date Information: month, day_of_week.
# Setup and Installation
Prerequisites
- Python 3.8 or later
- MLflow installed
- Libraries: pandas, scikit-learn, xgboost, seaborn, matplotlib

# Project Workflow
1. Exploratory Data Analysis (EDA):

- Visualize distributions, correlations, and outliers.
- Tools: seaborn, matplotlib.
2. Preprocessing:

- Handle missing data, encode categorical features, and scale numerical ones.
- Techniques: LabelEncoding, OneHotEncoding, StandardScaler.
3. Model Training:

- Models used: Logistic Regression, Random Forest, XGBoost.
- Feature selection and hyperparameter tuning.
4. Evaluation:

- Metrics: Accuracy, Precision, Recall, ROC-AUC.
5. Experiment Tracking:

- Log parameters, metrics, and artifacts in MLflow.
6. Model Deployment:

- Save the best-performing model for deployment.
