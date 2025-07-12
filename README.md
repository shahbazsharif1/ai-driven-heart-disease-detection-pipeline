<h1 align="center">🫀 Heart Disease Prediction using Machine Learning</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python" />
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License" />
  <img src="https://img.shields.io/badge/Made%20with-%E2%9D%A4-red" alt="Love" />
</p>

---

## 📌 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Techniques Used](#techniques-used)
- [Results](#results)
- [Requirements](#requirements)
- [References](#references)

---

## 📚 Overview
This project predicts the presence of heart disease using various machine learning algorithms. The dataset contains patient medical attributes, and the target variable indicates whether a patient has heart disease.

---

## 🏥 Dataset
- **Size:** 968 observations, 14 features after cleaning (originally 1025).
- **Target:** `1` indicates presence of heart disease, `0` absence.

---

## ⚙️ Techniques Used
- **Data Cleaning:** Outlier removal, renaming columns
- **Feature Engineering:** Label encoding & one-hot encoding
- **Feature Scaling:** StandardScaler

**Model Building:**
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Gaussian Naive Bayes

- **Hyperparameter Tuning:** GridSearchCV
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, ROC AUC

---

## 🚀 Results

| Model                  | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|-------------------------|----------|-----------|--------|----------|---------|
| **Random Forest**       | 96.7%    | 95.4%     | 97.9%  | 96.7%    | 0.996   |
| Decision Tree           | 93.8%    | 93.9%     | 93.3%  | 93.3%    | 0.964   |
| Support Vector Machine  | 88.9%    | 84.0%     | 95.3%  | 89.3%    | 0.958   |
| Logistic Regression     | 83.4%    | 79.2%     | 89.3%  | 83.9%    | 0.916   |
| Gaussian Naive Bayes    | 81.2%    | 78.6%     | 83.9%  | 81.1%    | 0.887   |

✅ Hyperparameter tuning improved:
- **SVC F1:** 89% ➜ **96%**
- **Logistic Regression F1:** 83% ➜ **88%**

---

## 🛠 Requirements
- Python 3.8+
- Libraries:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn

---

## ✍️ References
- Shah et al. (2020). *Heart disease prediction using machine learning techniques.*
- Ramalingam et al. (2018). *Heart disease prediction using machine learning techniques: A survey.*

---

<p align="center">
  <b>🚀 Built with passion & machine learning 🚀</b>
</p>
