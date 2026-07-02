# Predicting Customer Insurance Purchase Behaviour: A Comparative Study of Classification Algorithms

**Mini Project Report — AI in Deep Learning Internship Training Programme**
**Project I — Customer Insurance Purchases Case Study**

**Submitted by:** Abbas Lokhandwala
**Date of Submission:** *4th July 2026*

---

## Abstract

This project addresses a binary classification problem for a bank-insurance company: predicting whether a customer will purchase health insurance based on two attributes, Age and Estimated Salary. Using the publicly available Social_Network_Ads dataset (400 customer records), five supervised classification algorithms — Logistic Regression, K-Nearest Neighbours (KNN), Support Vector Machine (SVM) with an RBF kernel, Decision Tree, and Random Forest — were implemented, trained, and evaluated on a held-out test set (25% of the data, stratified). Model performance was compared using accuracy, precision, recall, F1-score, and the gap between training and test accuracy as an indicator of overfitting. KNN achieved the highest test accuracy (90%) with a modest train-test gap (3.7 percentage points), while Random Forest showed the largest overfitting gap (13.7 points) despite near-perfect training accuracy. Decision boundaries were visualised for every model, and the best-performing model was used to predict outcomes for eight specified customer scenarios and to test three hypotheses about the relationship between age, salary, and purchase likelihood. The analysis found that Age (correlation 0.62) is a stronger individual predictor of purchase behaviour than Estimated Salary (correlation 0.36), and that non-linear models (KNN, SVM) outperform the linear Logistic Regression baseline, indicating the true decision boundary is curved rather than a straight line.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Literature Review](#2-literature-review)
3. [Problem Statement](#3-problem-statement)
4. [Data Collection and Preprocessing](#4-data-collection-and-preprocessing)
5. [Methodology](#5-methodology)
6. [Implementation](#6-implementation)
7. [Results](#7-results)
8. [Discussion](#8-discussion)
9. [Conclusion](#9-conclusion)
10. [References](#10-references)
11. [Appendices](#11-appendix-a--additional-decision-boundary-plots)

---

## 1. Introduction

Insurance companies face a persistent business challenge: identifying which prospective customers are most likely to purchase a policy, so that marketing effort and budget can be targeted efficiently rather than spread thinly across the entire customer base. This project simulates that scenario for a bank-insurance company using two easily-collected, non-sensitive customer attributes — Age and Estimated Salary — to predict a binary outcome: whether the customer purchases insurance (1) or not (0).

The objective of this project is not to build a single model, but to conduct a comparative study of five widely-used classification algorithms, evaluate them on consistent metrics, visualise their decision boundaries, and use the best-performing model to answer specific business questions and test behavioural hypotheses about customers.

### 1.1 Objectives

- Preprocess and explore a real customer dataset (Age, Estimated Salary, Purchased).
- Implement five classification algorithms: Logistic Regression, KNN, SVM, Decision Tree, and Random Forest.
- Evaluate and compare their accuracy, precision, recall, F1-score, and overfitting behaviour.
- Visualise decision boundaries for each algorithm.
- Predict purchase outcomes for specified customer age/salary scenarios.
- Formulate and statistically test hypotheses about age and salary as purchase drivers.

---

## 2. Literature Review

Binary classification on demographic attributes is one of the most common applied machine-learning tasks in the insurance and marketing analytics domain. Logistic Regression has traditionally been the default baseline for such problems because of its interpretability and the direct probabilistic output it provides, but it assumes a linear decision boundary in feature space, which limits its accuracy when the true relationship is non-linear.

K-Nearest Neighbours and kernel-based Support Vector Machines are commonly used to capture non-linear boundaries: KNN makes no assumption about the underlying data distribution and instead classifies a point by the majority vote of its nearest neighbours, while SVM with an RBF kernel projects the data into a higher-dimensional space where a linear separator corresponds to a non-linear boundary in the original feature space. Tree-based methods — Decision Trees and their ensemble extension, Random Forest — partition the feature space into axis-aligned regions and are popular for their interpretability (Decision Tree) or their robustness and reduced variance (Random Forest), though ensembles are more prone to overfitting the training data if not regularised (for example by limiting tree depth or the number of estimators).

Prior applied studies using the same Age/EstimatedSalary/Purchased dataset structure (commonly seen in introductory machine learning coursework, e.g. the Social Network Ads dataset) consistently report that non-linear models such as KNN and kernel SVM outperform plain Logistic Regression on this data, which motivated the comparative approach adopted in this project.

---

## 3. Problem Statement

Given a customer's Age and Estimated Salary, predict whether the customer will purchase health insurance (Purchased = 1) or not (Purchased = 0). The problem is framed as a supervised binary classification task.

### 3.1 Assumptions and Limitations

- Only Age and Estimated Salary are used as predictive features; other potentially relevant factors (e.g. existing medical conditions, family size, occupation) are not available in this dataset and are therefore outside the scope of this study.
- The dataset (400 records) is assumed to be representative of the broader customer population; results may not generalise to salary ranges far outside the observed training range (₹15,000–₹150,000), which is directly tested in Question 2 of this report.
- Class distribution is imbalanced (257 non-purchasers vs 143 purchasers, roughly 64%/36%), which is accounted for using a stratified train-test split.

---

## 4. Data Collection and Preprocessing

The dataset used is the Social_Network_Ads dataset, containing 400 customer records with the columns User ID, Gender, Age, EstimatedSalary, and Purchased. For this study, only Age and EstimatedSalary were used as input features, and Purchased as the target label, consistent with the business goal of excluding personal/identifying data (User ID, Gender were not used as predictors).

### 4.1 Preprocessing Steps

- **Feature selection:** retained Age and EstimatedSalary as the two predictive features; dropped User ID and Gender.
- **Train-test split:** 75% training (300 records) / 25% testing (100 records), stratified on the target to preserve the class ratio in both sets.
- **Feature scaling:** applied StandardScaler (zero mean, unit variance) to Age and EstimatedSalary, since KNN and SVM are distance-based and sensitive to feature scale; the same scaler fitted on the training set was used to transform the test set and all later prediction scenarios, to avoid data leakage.

### 4.2 Exploratory Data Analysis

The scatter plot below shows the raw relationship between Age, Estimated Salary, and purchase decision. A visible pattern emerges: older customers and/or customers with higher estimated salaries are more likely to have purchased insurance (green points), while younger customers with mid-range salaries tend not to purchase (red points).

![Age vs Estimated Salary](outputs/01_eda_scatter.png)
*Figure 1: Age vs Estimated Salary coloured by purchase decision*

---

## 5. Methodology

Five classification algorithms were selected to represent the major families of classifiers used in applied machine learning: a linear model, a distance-based model, a margin-based kernel model, a single tree model, and an ensemble-of-trees model.

| Algorithm | Type | Key Hyperparameters Used |
|---|---|---|
| Logistic Regression | Linear model | Default L2 regularisation, random_state=0 |
| K-Nearest Neighbours | Instance-based | k = 5, Minkowski distance (p = 2, i.e. Euclidean) |
| Support Vector Machine | Kernel/margin-based | RBF kernel, probability=True, random_state=0 |
| Decision Tree | Tree-based | Criterion = entropy, max_depth = 4, random_state=0 |
| Random Forest | Ensemble (bagging of trees) | 100 estimators, criterion = entropy, random_state=0 |

Each model was trained on the scaled training data and evaluated on the scaled test data using: Accuracy (overall correctness), Precision (of predicted purchasers, how many actually purchased), Recall (of actual purchasers, how many were correctly identified), F1-score (harmonic mean of precision and recall), and the Train-Test Accuracy Gap (a proxy for overfitting: a large gap means the model memorised the training data rather than learning a generalisable pattern).

---

## 6. Implementation

The project was implemented in Python using pandas for data handling, scikit-learn for modelling, and matplotlib/seaborn for visualisation. The full pipeline — data loading, preprocessing, model training, evaluation, decision-boundary plotting, and scenario prediction — is contained in a single reproducible script (`analysis.py`), so the complete project can be re-run end-to-end.

**GitHub Repository:** *(add your repo link here)*

### 6.1 Pipeline Summary

- Load `Social_Network_Ads.csv` with pandas.
- Select Age and EstimatedSalary as X, Purchased as y.
- Split into train/test sets (75/25, stratified, random_state = 0).
- Fit StandardScaler on training data; transform train and test sets.
- Train all five classifiers on the scaled training data.
- Evaluate each on the test set; store accuracy, precision, recall, F1-score.
- Plot decision boundaries for each model over the test set.
- Use the best model (by test accuracy) to predict the specified scenarios and test the hypotheses.

---

## 7. Results

### 7.1 Comparative Performance of All Algorithms

| Algorithm | Train Acc. | Test Acc. | Precision | Recall | F1-Score | Overfit Gap |
|---|---|---|---|---|---|---|
| KNN (k=5) | 93.7% | **90.0%** | 0.842 | 0.889 | 0.865 | 3.7 pts |
| SVM (RBF kernel) | 91.7% | 89.0% | 0.805 | 0.917 | 0.857 | 2.7 pts |
| Decision Tree | 94.0% | 88.0% | 0.816 | 0.861 | 0.838 | 6.0 pts |
| Random Forest | 99.7% | 86.0% | 0.789 | 0.833 | 0.811 | 13.7 pts |
| Logistic Regression | 85.3% | 81.0% | 0.793 | 0.639 | 0.708 | 4.3 pts |

![Algorithm Comparison](outputs/02_comparison_bar.png)
*Figure 2: Train accuracy, test accuracy, and F1-score across all five algorithms*

KNN achieved the best balance of test accuracy (90%) and low overfitting gap (3.7 points), making it the selected "optimal" algorithm for this dataset per the project's evaluation criterion (high accuracy without excessive overfitting). SVM (RBF) was a close second with the highest recall (91.7%), meaning it missed the fewest actual purchasers. Random Forest, despite scoring almost perfectly on training data (99.7%), generalised the worst of the five models (largest overfit gap of 13.7 points) — a classic sign of an ensemble memorising the training set rather than learning the underlying pattern. Logistic Regression, the only strictly linear model, had the lowest recall (63.9%), confirming that the true boundary between purchasers and non-purchasers is curved rather than a straight line.

### 7.2 Decision Boundaries

Decision boundary plots (generated for every model) visualise how each algorithm partitions the Age vs Estimated Salary space into "will purchase" and "will not purchase" regions.

![KNN Decision Boundary](outputs/04_boundary_knn.png)
*Figure 3: KNN decision boundary (test set) — curved, adapts closely to local data density*

![Logistic Regression Decision Boundary](outputs/03_boundary_logistic.png)
*Figure 4: Logistic Regression decision boundary (test set) — a single straight line*

*(Remaining boundary plots for SVM, Decision Tree, and Random Forest are in the `outputs/` folder.)*

### 7.3 Graphical Analysis and Predictions — Question Set 1

Using the trained models (with the KNN model as the primary reference, since it was the most accurate), predictions were generated for the four specified scenarios. "No Salary" was interpreted as EstimatedSalary = 0.

| Age | Salary | Log. Reg. | KNN | SVM | Dec. Tree | Rand. Forest |
|---|---|---|---|---|---|---|
| 30 | ₹87,000 | No (0) | No (0) | No (0) | No (0) | No (0) |
| 40 | No Salary (0) | No (0) | No (0) | Yes (1) | No (0) | No (0) |
| 40 | ₹100,000 | Yes (1) | Yes (1) | Yes (1) | Yes (1) | Yes (1) |
| 50 | No Salary (0) | No (0) | Yes (1) | Yes (1) | Yes (1) | Yes (1) |

Four of five models agree that a 30-year-old on ₹87,000 salary will not purchase, and all five agree that a 40-year-old on ₹100,000 will purchase. The models disagree most on the two zero-salary scenarios, showing that when salary information is missing/zero, Age alone starts to dominate the prediction — different algorithms weight that signal differently.

### 7.4 Graphical Analysis and Predictions — Question Set 2 (Extreme Salary Values)

The training data's EstimatedSalary values range only from ₹15,000 to ₹150,000. Question Set 2 asks for predictions at salary figures that are 4x to over 600,000x outside that range.

| Age | Salary | Log. Reg. | KNN | SVM | Dec. Tree | Rand. Forest |
|---|---|---|---|---|---|---|
| 18 | No Salary (0) | No (0) | No (0) | No (0) | No (0) | No (0) |
| 22 | ₹6,00,000 | Yes (1) | Yes (1) | Yes (1) | Yes (1) | Yes (1) |
| 35 | ₹25,00,000 | Yes (1) | Yes (1) | Yes (1) | Yes (1) | Yes (1) |
| 60 | ₹10,00,00,000 | Yes (1) | Yes (1) | Yes (1) | Yes (1) | Yes (1) |

All five models unanimously predict "will purchase" for every scenario with a non-zero extreme salary, regardless of age. This is important to flag: once the standardised salary value is far outside the training range, every model is **extrapolating** rather than interpolating, and its prediction should not be trusted with the same confidence as predictions made within the observed ₹15,000–₹150,000 range.

### 7.5 Hypothesis Testing

Three hypotheses were tested using the best-performing model (KNN), by holding one feature fixed and varying the other.

**Hypothesis 1: Younger individuals with higher salaries are more likely to purchase than younger individuals with lower salaries.**

| Age | Salary | Predicted | P(Purchase) |
|---|---|---|---|
| 25 | ₹30,000 | No (0) | 0.00 |
| 25 | ₹1,20,000 | Yes (1) | 0.80 |
| 45 | ₹30,000 | Yes (1) | 1.00 |
| 45 | ₹1,20,000 | Yes (1) | 1.00 |

*Result:* Supported for the younger group — at age 25, raising salary from ₹30,000 to ₹1,20,000 flips the prediction from No to Yes. However, at age 45 the model predicts Yes regardless of salary, showing salary matters more at younger ages than older ones (an interaction effect, not a simple additive one).

**Hypothesis 2: At a fixed salary, older customers are more likely to purchase.**

| Age | Salary | Predicted | P(Purchase) |
|---|---|---|---|
| 20 | ₹80,000 | No (0) | 0.00 |
| 30 | ₹80,000 | No (0) | 0.00 |
| 40 | ₹80,000 | No (0) | 0.00 |
| 50 | ₹80,000 | Yes (1) | 1.00 |
| 60 | ₹80,000 | Yes (1) | 0.80 |

*Result:* Supported. Holding salary constant at ₹80,000, the prediction flips from No to Yes between age 40 and age 50, confirming a strong, fairly sharp age effect around the mid-40s.

**Hypothesis 3: At a fixed age, higher salary increases purchase likelihood.**

| Age | Salary | Predicted | P(Purchase) |
|---|---|---|---|
| 35 | ₹20,000 | No (0) | 0.00 |
| 35 | ₹50,000 | No (0) | 0.00 |
| 35 | ₹80,000 | No (0) | 0.00 |
| 35 | ₹1,10,000 | No (0) | 0.40 |
| 35 | ₹1,50,000 | Yes (1) | 1.00 |

*Result:* Supported. Holding age constant at 35, probability of purchase rises steadily from 0.00 to 1.00 as salary increases from ₹20,000 to ₹1,50,000, confirming salary is positively associated with purchase likelihood, though the effect only becomes decisive at the higher end of the salary range.

Additionally, a simple correlation check against the target variable found Age (r = 0.62) to be a noticeably stronger individual predictor than EstimatedSalary (r = 0.36) — this refines the third example hypothesis given in the assignment brief ("salary might have a stronger impact than age"): in this dataset, the opposite is true, though Hypothesis 1 shows the two features interact rather than act independently.

---

## 8. Discussion

The results show a clear trade-off across the five algorithms. Non-linear, flexible models (KNN, SVM) captured the curved boundary between purchasers and non-purchasers more accurately than the linear Logistic Regression baseline, which is consistent with the visible pattern in the exploratory scatter plot (Figure 1), where the purchase boundary bends rather than running straight.

Random Forest's near-perfect training accuracy (99.7%) alongside its lowest test accuracy (86%) is the clearest overfitting signal in this study: with only two features and 300 training rows, 100 unconstrained trees can memorise noise in the training set rather than learning the general Age/Salary pattern. This is a useful, generalisable lesson: more model complexity is not automatically better, and the train-test accuracy gap is a more honest indicator of a model's real-world reliability than training accuracy alone.

A limitation of this study is that Age and EstimatedSalary are the only two features available; a production insurance-recommendation system would likely also benefit from features such as existing dependents, health status, or occupation, which were not present in this dataset. The extreme-value scenarios in Question Set 2 also highlight a broader machine-learning caution: models are reliable when interpolating within the range of their training data, but predictions on far-out-of-range inputs (extrapolation) should be treated with much lower confidence.

---

## 9. Conclusion

This project implemented and compared five classification algorithms — Logistic Regression, KNN, SVM, Decision Tree, and Random Forest — to predict customer insurance purchase behaviour from Age and Estimated Salary. KNN (k=5) was identified as the best-suited algorithm for this dataset, combining the highest test accuracy (90%) with a modest, healthy train-test gap (3.7 points), while Random Forest, despite the highest training accuracy, overfit the most and generalised the worst. The best model was used to predict eight specified customer scenarios and to test three hypotheses, confirming that both Age and Salary drive purchase likelihood, with Age emerging as the individually stronger predictor and the two features showing a clear interaction effect. This satisfies the business goal of identifying a model that balances precision and generalisation for the bank-insurance company's customer targeting problem.

### 9.1 Real-Life Applications

- **Insurance/BFSI cross-selling:** A bank can score its existing savings-account customers with a similar Age/Salary (and other available) feature set to prioritise outreach for a new insurance or investment product, focusing the sales team's limited time on the highest-probability leads rather than contacting the entire customer base.
- **EdTech/HR early-warning systems:** The same workflow (compare multiple classifiers, check overfitting via train-test gap, visualise decision boundaries) can be reused to predict student dropout risk or employee attrition risk from a handful of easily-available features (e.g. attendance/engagement score and tenure), so that a college or company can intervene early with the students/employees flagged as highest-risk.

### 9.2 Lessons Learned

- A model with the highest training accuracy is not automatically the best model — always check the train-test gap before trusting a model (Random Forest here vs KNN).
- Feature scaling matters for distance-based and margin-based models (KNN, SVM) — skipping it would let EstimatedSalary (large numbers) dominate Age (small numbers) purely due to scale, not real importance.
- Predictions far outside the training data's feature range (extrapolation) should be treated cautiously, regardless of which algorithm is used.
- Simple correlation checks are a fast sanity check to validate or challenge intuitive hypotheses before formal model-based testing.

---

## 10. References

1. Pedregosa, F. et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
2. Cortes, C. and Vapnik, V., "Support-Vector Networks," *Machine Learning*, vol. 20, no. 3, pp. 273–297, 1995.
3. Breiman, L., "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.
4. Cover, T. and Hart, P., "Nearest Neighbor Pattern Classification," *IEEE Transactions on Information Theory*, vol. 13, no. 1, pp. 21–27, 1967.
5. Social Network Ads Dataset, publicly available machine-learning practice dataset (Age, EstimatedSalary, Purchased), used under standard open dataset practice for educational classification exercises.

---

## 11. Appendix A — Additional Decision Boundary Plots

![SVM Decision Boundary](outputs/05_boundary_svm.png)
*Figure A1: SVM (RBF kernel) decision boundary*

![Decision Tree Decision Boundary](outputs/06_boundary_decision.png)
*Figure A2: Decision Tree decision boundary*

![Random Forest Decision Boundary](outputs/07_boundary_random.png)
*Figure A3: Random Forest decision boundary*

## Appendix B — How to Reproduce This Project

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python analysis.py
```

All plots and CSV result tables are written to an `outputs/` folder automatically.
