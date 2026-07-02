# 🏦 Customer Insurance Purchase Prediction — Classification Algorithm Comparison

A machine learning mini-project that predicts whether a bank's customer will purchase health insurance based on their **Age** and **Estimated Salary**, using and comparing **five classification algorithms**.

> Mini Project | AI in Deep Learning — Internship Training Programme
> **Author:** Abbas

---

## 📌 Problem Statement

A Bank Insurance Company wants to predict whether a new customer will purchase health insurance, using only their **Age** and **Estimated Salary** — without using any private/personal data (no passwords, account numbers, etc.).

This project builds and compares **five classification algorithms** to find the one that best balances **accuracy** and **generalisation** (i.e. performs well without overfitting).

---

## 📊 Dataset

- **File:** `Social_Network_Ads.csv`
- **Rows:** 400 customer records
- **Columns used:** `Age`, `EstimatedSalary` (features) → `Purchased` (target: 0 = No, 1 = Yes)
- **Columns dropped:** `User ID`, `Gender` (not used as predictors, per business requirement to exclude non-essential personal data)

---

## 🧠 Algorithms Compared

| # | Algorithm | Type |
|---|---|---|
| 1 | Logistic Regression | Linear model |
| 2 | K-Nearest Neighbours (KNN) | Instance-based |
| 3 | Support Vector Machine (SVM, RBF kernel) | Kernel-based |
| 4 | Decision Tree | Tree-based |
| 5 | Random Forest | Ensemble of trees |

---

## ⚙️ Methodology

1. **Data Preprocessing** — feature selection, 75/25 stratified train-test split, `StandardScaler` feature scaling
2. **Model Training** — all 5 algorithms trained on the same scaled training data
3. **Evaluation** — Accuracy, Precision, Recall, F1-Score, and Train-Test Accuracy Gap (overfitting check)
4. **Visualization** — decision boundary plots for every model
5. **Scenario Prediction** — predictions for 8 specific customer profiles given in the assignment (including extreme salary values to test extrapolation behaviour)
6. **Hypothesis Testing** — tested 3 hypotheses about how Age and Salary drive purchase likelihood

---

## 🏆 Results Summary

| Algorithm | Train Acc. | Test Acc. | F1-Score | Overfit Gap |
|---|---|---|---|---|
| **KNN (k=5)** | 93.7% | **90.0%** | 0.865 | 3.7 pts ✅ |
| SVM (RBF kernel) | 91.7% | 89.0% | 0.857 | 2.7 pts |
| Decision Tree | 94.0% | 88.0% | 0.838 | 6.0 pts |
| Random Forest | 99.7% | 86.0% | 0.811 | 13.7 pts ⚠️ |
| Logistic Regression | 85.3% | 81.0% | 0.708 | 4.3 pts |

**🥇 Best Model: KNN (k=5)** — highest test accuracy with the healthiest train-test gap, meaning it generalises well to unseen customers instead of just memorising the training data.

**Key Insight:** Random Forest scored almost perfectly on training data (99.7%) but dropped the most on test data — a classic sign of **overfitting**. Higher training accuracy does not mean a better model.

**Feature Importance:** `Age` (correlation = 0.62) is a stronger individual predictor of purchase behaviour than `EstimatedSalary` (correlation = 0.36).

---

## 📁 Repository Structure

```
├── analysis.py                  # Full ML pipeline (data → models → evaluation → plots → predictions)
├── Social_Network_Ads.csv       # Dataset (400 records)
├── outputs/                     # Auto-generated on running the script
│   ├── 01_eda_scatter.png       # Exploratory data analysis plot
│   ├── 02_comparison_bar.png    # Algorithm comparison chart
│   ├── 03-07_boundary_*.png     # Decision boundary plots (all 5 models)
│   ├── comparison_table.csv     # Full metrics table
│   ├── q1_predictions.csv       # Predictions for Question Set 1
│   ├── q2_predictions.csv       # Predictions for Question Set 2 (extreme values)
│   └── h1/h2/h3_results.csv     # Hypothesis test results
├── Project_I_Insurance_Classification_Report.docx   # Full IEEE-format project report
└── README.md
```

---

## 🚀 How to Run

**Requirements:** Python 3.10+

```bash
# 1. Clone this repository
git clone <your-repo-link>
cd <repo-folder>

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# 3. Run the analysis
python analysis.py
```

All graphs and result tables will be generated automatically inside an `outputs/` folder.

---

## 🛠️ Tech Stack

- **Python 3** — core language
- **pandas / numpy** — data handling
- **scikit-learn** — model building & evaluation
- **matplotlib / seaborn** — data visualization

---

## 📚 Key Learnings

- A model with the **highest training accuracy is not automatically the best model** — always check the train-test gap before trusting it.
- **Feature scaling matters** for distance-based models like KNN and SVM.
- Predictions on inputs **far outside the training data's range** (extrapolation) should be treated with caution, regardless of the algorithm used.
- Non-linear models (KNN, SVM) outperformed the linear baseline (Logistic Regression) here, showing the true decision boundary is curved rather than a straight line.

---

## 📬 Contact

**Abbas Lokhandwala** — MIT School of Computing, MIT-ADT University, Pune
