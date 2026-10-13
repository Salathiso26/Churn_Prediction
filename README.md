# Bank Customer Churn Prediction

An end-to-end machine learning project that predicts which bank customers are likely to leave, explains each prediction, monitors data drift, and serves results through an API and an interactive dashboard. The whole project runs from a single script, `churn_app.py`.

## Overview

Customer churn is costly for banks, and keeping a customer is usually cheaper than winning a new one. This project goes beyond a single trained model. It covers data checks, feature engineering, model comparison, explainability, drift monitoring, automatic retraining, serving and testing.

## Dataset

The project uses the Bank Customer Churn Modelling dataset (`Churn_Modelling.csv`), available on Kaggle:
https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling

It contains 10,000 bank customers with no missing values or duplicates. The target is `Exited` (1 = the customer left). About 20% of customers churned (2,037 of 10,000).

Input features: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary.

Data split: 6,400 train, 1,600 validation, 2,000 test.

Findings from the exploratory analysis:

| Segment | Churn rate |
|---|---|
| France / Spain | 16.2% / 16.7% |
| Germany | 32.4% |
| 1 product | 27.7% |
| 2 products | 7.6% |
| 3 products | 82.7% |
| 4 products | 100% |
| Active members | 14.3% |
| Inactive members | 26.9% |
| Female / Male | 25.1% / 16.5% |

## Features

- Data checks and exploratory analysis
- Engineered features (for example ProductsPerTenure and TenureByAge)
- Model comparison: logistic regression, random forest and LightGBM
- Class weighting to handle the imbalance between churners and non-churners
- Evaluation with ROC-AUC, PR-AUC, recall, precision, F1 and Brier score
- Experiment tracking and model registry with MLflow (optional)
- Per-customer explanations with SHAP
- REST API built with FastAPI
- Interactive dashboard built with Streamlit
- Smoke tests

## Results

Model comparison on the held-out test set:

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Logistic regression | 0.7769 | 0.4706 | 0.4965 | 0.3846 | 0.7002 |
| Random forest | 0.8536 | 0.6938 | 0.5737 | 0.8080 | 0.4447 |
| LightGBM | 0.8555 | 0.6992 | 0.6212 | 0.5551 | 0.7052 |

LightGBM was selected as the final model. It has the best ROC-AUC, PR-AUC and F1, and it identifies about 70% of the customers who actually leave. The random forest is more precise but misses more than half of the churners, which matters more when the goal is retention.

Final model on the test set (threshold 0.5): ROC-AUC 0.8555, PR-AUC 0.6992, recall 0.7052, F1 0.6212, precision 0.5551, Brier score 0.1249.

The strongest churn drivers according to SHAP are the number of products, age, account activity and balance.

## Project structure

```
Churn_Prediction_Bank/
  churn_app.py          Full pipeline and all commands
  Churn_Modelling.csv   Dataset (download from Kaggle, see below)
  README.md
  
```

## Dashboard

The dashboard has two tabs:

- **Single customer:** adjust a customer profile and see the churn probability, the risk band (low, medium or high) and the top SHAP drivers, shown as a table and a chart.
- **Portfolio and drift:** portfolio-level results and drift status.
<img width="1365" height="712" alt="Screenshot 2026-10-13 172422" src="https://github.com/user-attachments/assets/d0482e05-73af-4551-96fb-c47f4d596ba7" />
<img width="1360" height="480" alt="Screenshot 2026-10-13 172435" src="https://github.com/user-attachments/assets/095286fe-39b0-4769-9eee-02203c4e5f7c" />

<img width="1355" height="423" alt="Screenshot 2026-10-13 172448" src="https://github.com/user-attachments/assets/e7cdc9a1-2042-45c5-9a3b-4a7eefe3591b" />

## Drift monitoring

Each feature is compared against the training data using PSI and the Kolmogorov-Smirnov test.

| PSI | Status |
|---|---|
| below 0.10 | Stable |
| 0.10 to 0.25 | Moderate shift, watch |
| 0.25 or above | Major shift, retrain |

`auto-retrain` retrains the model only when the worst feature crosses the 0.25 threshold.

## Tech stack

Python, pandas, NumPy, scikit-learn, LightGBM, SHAP, SciPy, MLflow, FastAPI, Uvicorn, Streamlit


## Dataset source and acknowledgement

The data comes from the Bank Customer Churn Modelling dataset on Kaggle:
https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling

Please check the licence and terms on the Kaggle page before reusing or redistributing the data.

## Limitations

- **Drift is simulated.** The Kaggle dataset is static and has no timestamps, so `simulate-drift` injects a shift (older, higher-balance, lower-credit customers) to demonstrate the retraining logic. Real drift would come from live data.
- **Probabilities are not calibrated.** Balanced class weights push probabilities upward, so treat the output as a risk score for ranking customers and not as an exact chance of leaving.
- **Retraining uses the original dataset.** `auto-retrain` refits on the same data. A production version would add the newly labelled data.
- **No promotion gate.** A retrained model is registered without being compared against the current one.
- **Single script.** There is no Airflow, Docker or CI. `auto-retrain` can be run on a schedule with cron or Windows Task Scheduler.

## Possible next steps

- Champion-challenger check before a new model version is promoted
- Probability calibration
- Persist predictions and drift reports so the dashboard can show drift over time
- Dockerfile and GitHub Actions workflow
- Airflow DAG that calls the same functions
  
## Author

Salathiso Ntsele
LinkedIn:https://www.linkedin.com/in/salathiso-ntsele/
