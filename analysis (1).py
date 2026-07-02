"""
Project I - Customer Insurance Purchases Case Study
Comparative analysis of classification algorithms
CodeHarvest / IEEE TechForGood style internship mini-project
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

OUT = "/home/claude/project/outputs"
import os
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
df = pd.read_csv("/home/claude/project/Social_Network_Ads.csv")
print("Shape:", df.shape)
print(df.head())
print(df['Purchased'].value_counts())

X = df[['Age', 'EstimatedSalary']].values
y = df['Purchased'].values

# ---------------------------------------------------------------
# 2. EDA - scatter of raw data
# ---------------------------------------------------------------
plt.figure(figsize=(7,5))
mask0 = df['Purchased']==0
mask1 = df['Purchased']==1
plt.scatter(df.loc[mask0,'Age'], df.loc[mask0,'EstimatedSalary'], c='crimson', edgecolor='k', alpha=0.8, label='No (0)')
plt.scatter(df.loc[mask1,'Age'], df.loc[mask1,'EstimatedSalary'], c='seagreen', edgecolor='k', alpha=0.8, label='Yes (1)')
plt.title("Age vs Estimated Salary coloured by Purchase Decision")
plt.xlabel("Age"); plt.ylabel("Estimated Salary")
plt.legend(title="Purchased")
plt.tight_layout()
plt.savefig(f"{OUT}/01_eda_scatter.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 3. TRAIN/TEST SPLIT + SCALING
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

# ---------------------------------------------------------------
# 4. MODELS
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(random_state=0),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2),
    "SVM (RBF kernel)": SVC(kernel='rbf', probability=True, random_state=0),
    "Decision Tree": DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=0),
    "Random Forest": RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0),
}

results = []
trained = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    train_acc = accuracy_score(y_train, model.predict(X_train_sc))
    results.append({
        "Algorithm": name, "Train Accuracy": train_acc, "Test Accuracy": acc,
        "Precision": prec, "Recall": rec, "F1-Score": f1,
        "Overfit Gap": train_acc - acc
    })
    trained[name] = model
    print(f"\n{name}\nConfusion Matrix:\n{cm}\n{classification_report(y_test, y_pred)}")

results_df = pd.DataFrame(results).sort_values("Test Accuracy", ascending=False).reset_index(drop=True)
results_df.to_csv(f"{OUT}/comparison_table.csv", index=False)
print("\n=== COMPARISON TABLE ===")
print(results_df.to_string(index=False))

best_name = results_df.iloc[0]["Algorithm"]
best_model = trained[best_name]
print(f"\nBest model: {best_name}")

# ---------------------------------------------------------------
# 5. COMPARISON BAR CHART
# ---------------------------------------------------------------
plt.figure(figsize=(9,5))
x = np.arange(len(results_df))
plt.bar(x-0.2, results_df["Train Accuracy"], width=0.2, label="Train Acc")
plt.bar(x, results_df["Test Accuracy"], width=0.2, label="Test Acc")
plt.bar(x+0.2, results_df["F1-Score"], width=0.2, label="F1-Score")
plt.xticks(x, results_df["Algorithm"], rotation=20, ha='right')
plt.ylim(0,1.05)
plt.ylabel("Score")
plt.title("Algorithm Comparison: Accuracy & F1-Score")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/02_comparison_bar.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 6. DECISION BOUNDARY PLOTS (for every model, on test set)
# ---------------------------------------------------------------
def plot_decision_boundary(model, X_set_sc, y_set, title, fname):
    X1, X2 = np.meshgrid(
        np.arange(X_set_sc[:,0].min()-1, X_set_sc[:,0].max()+1, 0.02),
        np.arange(X_set_sc[:,1].min()-1, X_set_sc[:,1].max()+1, 0.02)
    )
    Z = model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
    plt.figure(figsize=(6.5,5))
    plt.contourf(X1, X2, Z, alpha=0.35, cmap=ListedColormap(('crimson','seagreen')))
    for cls, color in zip([0,1], ['crimson','seagreen']):
        plt.scatter(X_set_sc[y_set==cls,0], X_set_sc[y_set==cls,1],
                    c=color, edgecolor='k', s=25, label=f"Purchased={cls}")
    plt.title(title)
    plt.xlabel("Age (scaled)"); plt.ylabel("Estimated Salary (scaled)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

for i, (name, model) in enumerate(trained.items(), start=3):
    fname = f"{OUT}/{i:02d}_boundary_{name.split()[0].lower()}.png"
    plot_decision_boundary(model, X_test_sc, y_test, f"{name} — Decision Boundary (Test Set)", fname)

# ---------------------------------------------------------------
# 7. PREDICTIONS FOR SPECIFIED SCENARIOS  (Q1)
# ---------------------------------------------------------------
scenarios_q1 = pd.DataFrame({
    "Age":[30,40,40,50],
    "EstimatedSalary":[87000,0,100000,0]
})
scenarios_q1_sc = sc.transform(scenarios_q1)

print("\n=== Q1 SCENARIO PREDICTIONS (all models) ===")
q1_out = scenarios_q1.copy()
for name, model in trained.items():
    q1_out[name] = model.predict(scenarios_q1_sc)
print(q1_out.to_string(index=False))
q1_out.to_csv(f"{OUT}/q1_predictions.csv", index=False)

# ---------------------------------------------------------------
# 8. PREDICTIONS FOR SPECIFIED SCENARIOS (Q2 - extreme values)
# ---------------------------------------------------------------
scenarios_q2 = pd.DataFrame({
    "Age":[18,22,35,60],
    "EstimatedSalary":[0,600000,2500000,100000000]
})
scenarios_q2_sc = sc.transform(scenarios_q2)

print("\n=== Q2 SCENARIO PREDICTIONS (all models) ===")
q2_out = scenarios_q2.copy()
for name, model in trained.items():
    q2_out[name] = model.predict(scenarios_q2_sc)
print(q2_out.to_string(index=False))
q2_out.to_csv(f"{OUT}/q2_predictions.csv", index=False)

print("\nNote: EstimatedSalary in the original training data ranges from",
      df['EstimatedSalary'].min(), "to", df['EstimatedSalary'].max(),
      "-> Q2 salaries are far outside this range (extrapolation).")

# ---------------------------------------------------------------
# 9. HYPOTHESIS TESTING using best model
# ---------------------------------------------------------------
print("\n=== HYPOTHESIS TESTING (using", best_name, ") ===")

# H1: Younger + higher salary -> more likely to purchase
h1 = pd.DataFrame({"Age":[25,25,45,45], "EstimatedSalary":[30000,120000,30000,120000]})
h1_sc = sc.transform(h1)
h1['Predicted'] = best_model.predict(h1_sc)
h1['Prob_Purchase'] = best_model.predict_proba(h1_sc)[:,1] if hasattr(best_model,"predict_proba") else np.nan
print("\nH1 (age vs salary interaction):\n", h1.to_string(index=False))

# H2: Salary held constant, vary age
h2 = pd.DataFrame({"Age":[20,30,40,50,60], "EstimatedSalary":[80000]*5})
h2_sc = sc.transform(h2)
h2['Predicted'] = best_model.predict(h2_sc)
h2['Prob_Purchase'] = best_model.predict_proba(h2_sc)[:,1] if hasattr(best_model,"predict_proba") else np.nan
print("\nH2 (age effect at fixed salary=80000):\n", h2.to_string(index=False))

# H3: Age held constant, vary salary
h3 = pd.DataFrame({"Age":[35]*5, "EstimatedSalary":[20000,50000,80000,110000,150000]})
h3_sc = sc.transform(h3)
h3['Predicted'] = best_model.predict(h3_sc)
h3['Prob_Purchase'] = best_model.predict_proba(h3_sc)[:,1] if hasattr(best_model,"predict_proba") else np.nan
print("\nH3 (salary effect at fixed age=35):\n", h3.to_string(index=False))

h1.to_csv(f"{OUT}/h1_results.csv", index=False)
h2.to_csv(f"{OUT}/h2_results.csv", index=False)
h3.to_csv(f"{OUT}/h3_results.csv", index=False)

# Feature importance style check: correlation with target
corr = df[['Age','EstimatedSalary','Purchased']].corr()['Purchased']
print("\nCorrelation with Purchased:\n", corr)

print("\nDONE. All outputs saved to", OUT)
