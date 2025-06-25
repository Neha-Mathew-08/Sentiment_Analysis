#  Sentiment Analysis on Amazon Product Reviews

A machine learning project to classify customer product reviews as **positive** or **negative** using Natural Language Processing (NLP) techniques. The project includes data preprocessing, text vectorization (TF-IDF), model training, and performance evaluation, with a focus on handling class imbalance using **SMOTE**.

---

## Project Overview

This project applies supervised machine learning to sentiment analysis, training a **Logistic Regression** classifier on Amazon product review data. It demonstrates an end-to-end NLP workflow, from raw text preprocessing to balanced classification and interpretability using metrics and visualizations.

---

##  Dataset

- Source: [Kaggle – Consumer Reviews of Amazon Products](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products)
- Size: ~7,000 reviews used
- Preprocessing:
  - Removed neutral reviews (rating = 3)
  - Converted reviews with ratings ≥ 4 to positive (`label = 1`)
  - Converted ratings ≤ 2 to negative (`label = 0`)

---

##  Technologies Used

- **Language:** Python  
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `nltk`, `matplotlib`, `seaborn`  
- **Techniques:** TF-IDF vectorization, Logistic Regression, SMOTE (imbalanced-learn), evaluation metrics

---

##  Model Results

###  Before SMOTE (Imbalanced Dataset)

| Metric | Class 0 (Negative) | Class 1 (Positive) |
|--------|--------------------|--------------------|
| Precision | 0.56 | 0.98 |
| Recall    | 0.03 | 1.00 |
| F1-Score  | 0.06 | 0.99 |

- **Accuracy:** 97.7%  
- **Very poor recall for negative reviews**

---

### After SMOTE (Balanced Dataset)

| Metric | Class 0 (Negative) | Class 1 (Positive) |
|--------|--------------------|--------------------|
| Precision | 0.17 | 0.99 |
| Recall    | 0.64 | 0.93 |
| F1-Score  | 0.27 | 0.96 |

- **Accuracy:** 91.96%  
- **Significantly improved recall and F1-score for negative class**

---

##  Key Features

-  Cleaned and preprocessed noisy product reviews using NLTK
-  Transformed text data using **TF-IDF vectorization**
-  Trained and evaluated a **Logistic Regression classifier**
-  Handled class imbalance using **SMOTE oversampling**
-  Visualized performance using classification reports and confusion matrix

---

##  Author

**Neha Mathew**  
[GitHub](https://github.com/Neha-Mathew-08) | [LinkedIn](https://www.linkedin.com/in/neha-mathew-/)

