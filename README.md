# Credit Default Prediction with Deep Learning

Binary classification of next-month credit-card default using an Artificial Neural Network (ANN).

## Project goal

The goal is not only to maximize accuracy. The target is imbalanced, so the project focuses on correctly identifying the minority **default** class and evaluates the model with ROC-AUC, precision, recall, F1-score, and a confusion matrix.

## Dataset

**UCI Machine Learning Repository — Default of Credit Card Clients**

Dataset page:  
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

The dataset contains 30,000 observations and includes demographic information, credit limit, repayment status, bill amounts, payment amounts, and a binary next-month default target.

Download and extract:

`default of credit card clients.xls`

Place the `.xls` file in the same directory as the notebook.

## Workflow

1. Data quality checks
2. Target imbalance analysis
3. Stratified train / validation / test split
4. Leakage-safe preprocessing
5. One-hot encoding for `SEX`, `EDUCATION`, and `MARRIAGE`
6. Standardization of numeric features
7. Baseline ANN
8. Balanced class weights
9. Weighted ANN with Dropout and EarlyStopping
10. Threshold selection on validation data
11. Final evaluation on an untouched test set
12. ROC, Precision-Recall, and confusion-matrix visualizations

## Neural network

Architecture:

```text
Input
  ↓
Dense(32, ReLU)
  ↓
Dense(16, ReLU)
  ↓
Dropout(0.20)
  ↓
Dense(1, Sigmoid)
```

Loss: `binary_crossentropy`  
Optimizer: `Adam`  
Early stopping metric: validation ROC-AUC

## Why class weights?

The non-default class is much larger than the default class. A model can therefore achieve deceptively high accuracy while missing many default cases.

Balanced class weights increase the training penalty for mistakes on the minority class. This makes recall, precision, F1, and ROC-AUC more useful than accuracy alone.

## Important methodology choices

- The validation set is created **before** preprocessing.
- The preprocessor is fitted only on the training set.
- Class weights are calculated only from the training labels.
- Threshold selection is performed on validation data.
- The test set is kept untouched until the final evaluation.

These choices reduce data leakage and make the reported test performance more reliable.

## Environment

Recommended:

- Python 3.12
- TensorFlow 2.21
- scikit-learn
- pandas
- NumPy
- Matplotlib
- Seaborn

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
jupyter notebook
```

Open `credit_default_deep_learning.ipynb`, select the Python 3.12 DL kernel, and run the notebook from top to bottom.

## Repository structure

```text
.
├── credit_default_deep_learning.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

The dataset itself is not included in the repository. Download it from UCI using the link above.

## Notes

This is an educational portfolio project. In a production credit-risk system, threshold selection and model evaluation should also incorporate business costs, calibration, governance, stability monitoring, fairness analysis, and out-of-time validation.
