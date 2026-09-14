# Customer Churn Intelligence

> End-to-end customer churn prediction using an Artificial Neural Network (TensorFlow/Keras), Scikit-learn preprocessing, TensorBoard experiment tracking, and a Streamlit interface.

**Live demo:** https://customer-churn-intelligence-n6vahugcyi6w7btcsfwhvw.streamlit.app/

---

## Overview

Predicts whether a bank customer will churn (exit) based on their demographic and account profile. Built as a deep learning practical project — the goal was to implement an ANN end-to-end (data → training → deployment), not to benchmark against other ML algorithms. Tree-based model comparisons are planned as a separate project.

## Problem Statement

Customer churn directly affects recurring revenue and customer lifetime value. This project predicts `Exited ∈ {0, 1}` from customer attributes so a business could flag at-risk customers for retention outreach.

**Input:** Credit Score, Geography, Gender, Age, Tenure, Balance, Number of Products, Has Credit Card, Is Active Member, Estimated Salary
**Output:** Predicted churn probability + risk label

## Dataset

Bank customer dataset (`Churn_Modelling.csv`), ~10,000 records, 10 raw features + binary target `Exited`.

## Preprocessing

- Dropped irrelevant identifiers (`RowNumber`, `CustomerId`, `Surname`)
- `LabelEncoder` for `Gender`
- `OneHotEncoder` for `Geography` (avoids false ordinal relationship between countries)
- `StandardScaler` on all numeric features
- Encoders and scaler persisted as `.pkl` so inference uses the exact same transforms as training

## Model Architecture

```
Input (12 features)
      │
      ▼
Dense(64, ReLU)
      │
      ▼
Dense(32, ReLU)
      │
      ▼
Dense(1, Sigmoid)
      │
      ▼
Churn Probability
```

~2,945 trainable parameters. Sequential Keras model.

## Training

- Optimizer: Adam (lr = 0.01)
- Loss: Binary Crossentropy
- Metric: Accuracy
- Up to 100 epochs, `EarlyStopping` on `val_loss` (patience=10, restores best weights)
- Experiment tracking via TensorBoard (accuracy/loss curves per epoch)

## Results

~85–87% accuracy on held-out validation data.

## Why ANN (and not XGBoost/Random Forest)?

This project's purpose was learning to build a neural network end-to-end — architecture design, forward/backward propagation, callbacks, and deployment plumbing. ANN was chosen deliberately for that reason, not because it's guaranteed to beat tree-based models on tabular data. Baseline comparisons (Logistic Regression, Random Forest, XGBoost) are planned as a separate, dedicated ML-comparison project.

## Project Structure

```
customer-churn-intelligence/
├── data/
│   └── Churn_Modelling.csv
├── app.py                          # Streamlit inference app
├── experiments.ipynb               # data prep + ANN training pipeline
├── predictions.ipynb               # standalone inference walkthrough
├── model.h5                        # trained Keras ANN
├── scaler.pkl                      # fitted StandardScaler
├── label_encoder_gender.pkl        # fitted LabelEncoder (Gender)
├── one_hot_encoder_geography.pkl   # fitted OneHotEncoder (Geography)
├── requirements.txt
├── README.md
└── .gitignore
```

(`.venv/` and `logs/` exist locally for the virtual environment and TensorBoard runs, but are excluded from version control via `.gitignore`.)

## Running Locally

```bash
git clone https://github.com/Leonidus786/customer-churn-intelligence.git
cd customer-churn-intelligence
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack

Python · TensorFlow/Keras · Scikit-learn · Pandas · Streamlit · TensorBoard

## Limitations & Future Work

- Model output is a probability, not a calibrated guarantee — calibration is a separate evaluation step
- No baseline ML models evaluated yet (planned in a follow-up project)
- Single preprocessing pipeline (3 pickle files) could be consolidated into one `joblib` artifact

## Author

Built as part of a Deep Learning practical.
