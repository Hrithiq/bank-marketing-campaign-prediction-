# Bank Marketing Campaign Prediction

A machine learning project focused on predicting whether a customer will subscribe to a term deposit based on demographic, financial, and campaign-related features.

## Overview

Traditional bank marketing campaigns suffer from:
- Low conversion rates
- High operational costs
- Inefficient customer targeting

This project leverages **machine learning models** to identify high-potential customers and optimize marketing strategies.

---

## Objectives

- Predict customer subscription (Yes/No)
- Handle class imbalance using SMOTE
- Compare multiple classification models
- Improve recall for better customer targeting
- Provide actionable business insights

---

## Dataset

- **Source:** UCI Bank Marketing Dataset  
- **Records:** 45,211  
- **Features:** 16  
- **Target:** Subscription to term deposit (Yes/No)

---

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn

---

## Models Used

- Decision Tree
- Logistic Regression
- Naïve Bayes
- Support Vector Machine (SVM)

---

## Pipeline

1. Data Cleaning  
   - Replace "unknown" values  
   - Handle missing data  

2. Preprocessing  
   - Numerical: Scaling  
   - Categorical: One-hot encoding  

3. Class Imbalance Handling  
   - Applied **SMOTE** on training data  

4. Model Training  
   - GridSearchCV with 5-fold cross-validation  
   - Optimized for **F1-score**

---

## Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------|----------|----------|--------|----------|---------|
| SVM (SMOTE) | 0.856 | 0.438 | 0.804 | **0.567** | 0.914 |

### Best Model: SVM with SMOTE

- Achieved highest F1-score
- Doubled recall (from ~41% → ~80%)
- Strong classification performance on imbalanced data

---

## Key Insights

- **Call duration** is the strongest predictor  
- **Previous campaign success** significantly influences outcomes  
- **Seasonality (month)** impacts subscription rates  

### Business Impact:
- Better customer targeting
- Reduced campaign costs
- Improved conversion rates

---

## Trade-off

- Higher recall → Slightly lower precision  
- Acceptable in marketing:  
  > Missing a customer is worse than contacting an uninterested one

---

## Sample Output

*(Add confusion matrix image here if you want)*

---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bank-marketing-campaign-prediction.git ```

2. Install dependencies:
```bash
pip install -r requirements.txt```

3. Run the notebook:
```bash
jupyter notebook ```

## Contributors
- Hrithiq Gupta – SVM, Logistic Regression, SMOTE
- Arnav Sahu – Decision Tree
- Arnav Vijaywargiya – Naïve Bayes, Documentation

## Future Improvements
- Try Gradient Boosting (XGBoost, LightGBM)
- Deploy model using Flask/FastAPI
- Add explainability (SHAP values)
- Real-time prediction pipeline

## License

MIT License

---

# **requirements.txt**

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
jupyter
