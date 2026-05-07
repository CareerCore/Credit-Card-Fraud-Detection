# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using Logistic Regression.

## Overview

Credit card fraud is a major concern in the financial industry. This project builds a binary classification model to distinguish between **legitimate** and **fraudulent** transactions using a real-world dataset. To handle the severe class imbalance, under-sampling is applied before training.

## Dataset

- **File:** `creditcard.csv`
- **Target column:** `Class` (0 = Legitimate, 1 = Fraudulent)
- **Features:** Anonymized PCA-transformed features (`V1`–`V28`), `Time`, and `Amount`
- The dataset is heavily imbalanced — fraudulent transactions represent a very small fraction of all records.

## Project Workflow

1. **Data Loading & Exploration**
   - Load CSV into a Pandas DataFrame
   - Inspect structure, data types, and missing values
   - Analyze class distribution

2. **Exploratory Data Analysis**
   - Compare statistical summaries (`Amount`) for legit vs. fraud transactions
   - Group-wise mean comparison across both classes

3. **Handling Class Imbalance (Under-sampling)**
   - Sample 492 legitimate transactions (equal to the number of fraud cases)
   - Combine with all fraud transactions to create a balanced dataset (984 total)

4. **Model Training**
   - Split data: 80% training / 20% testing (stratified)
   - Train a **Logistic Regression** model

5. **Evaluation**
   - Measure accuracy on both training and test sets

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Model training & evaluation |

## Getting Started

### Prerequisites

```bash
pip install numpy pandas scikit-learn
```

### Run the Project

1. Place `creditcard.csv` in your working directory (or update the path in the script).
2. Run the notebook or script:

```bash
jupyter notebook credit_card_fraud_detection.ipynb
```

### Expected Output

```
Accuracy on Training data : ~0.94
Accuracy score on Test Data : ~0.92
```

*(Exact values may vary slightly due to random sampling)*

## Project Structure

```
├── creditcard.csv                        # Dataset
├── credit_card_fraud_detection.ipynb     # Main notebook
└── README.md                             # Project documentation
```

## Limitations & Future Improvements

- **Under-sampling** discards a large portion of legitimate transactions — consider **SMOTE** (over-sampling) as an alternative.
- Logistic Regression is a baseline; try **Random Forest**, **XGBoost**, or **Neural Networks** for better performance.
- Evaluate using **Precision**, **Recall**, and **F1-score** in addition to accuracy, as accuracy can be misleading on imbalanced data.
- Add a **confusion matrix** and **ROC-AUC curve** for richer evaluation.

## License

This project is open-source and available under the [MIT License](LICENSE).# Credit-Card-Fraud-Detection
AI-powered Credit Card Fraud Detection system built with Machine Learning to identify fraudulent transactions using data analysis and classification algorithms.
