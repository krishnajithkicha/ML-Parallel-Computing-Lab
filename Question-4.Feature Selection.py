# ======================================
# Feature Selection on Iris Dataset
# ======================================

# ---------- IMPORTS ----------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ======================================
# (a) LOAD DATASET AND SPLIT X, y
# ======================================
iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

print("Dataset shape:", X.shape)
print("Target classes:", iris.target_names)

# ======================================
# (b) EXPLORATORY DATA ANALYSIS (EDA)
# ======================================

# Basic statistics
print("\nDataset Description:")
print(X.describe())

# Class distribution
sns.countplot(x=y)
plt.title("Class Distribution")
plt.show()

# Pairplot
sns.pairplot(pd.concat([X, pd.Series(y, name="species")], axis=1),
             hue="species")
plt.show()

''' # Correlation heatmap
sns.heatmap(X.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
'''
# ======================================
# (c) FEATURE SELECTION TECHNIQUES
# ======================================

# ---- a) Univariate Feature Selection ----
selector = SelectKBest(score_func=f_classif, k=2)
X_uni = selector.fit_transform(X, y)
features_uni = X.columns[selector.get_support()]
print("\nUnivariate selected features:", list(features_uni))

# ---- b) Random Forest Feature Importance ----
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

rf_importance = pd.Series(rf.feature_importances_, index=X.columns)
print("\nRandom Forest Feature Importance:")
print(rf_importance.sort_values(ascending=False))

# ---- c) Recursive Feature Elimination (RFE) using SVM ----
svm_linear = SVC(kernel="linear")
rfe = RFE(estimator=svm_linear, n_features_to_select=2)
X_rfe = rfe.fit_transform(X, y)
features_rfe = X.columns[rfe.support_]
print("\nRFE selected features:", list(features_rfe))

# ======================================
# (d) MODEL EVALUATION USING SVM
# ======================================

# ---- Using ALL features ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

svm_all = SVC()
svm_all.fit(X_train, y_train)
y_pred_all = svm_all.predict(X_test)
acc_all = accuracy_score(y_test, y_pred_all)

# ---- Using SELECTED features (RFE) ----
X_selected = X[features_rfe]

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

svm_sel = SVC()
svm_sel.fit(X_train, y_train)
y_pred_sel = svm_sel.predict(X_test)
acc_sel = accuracy_score(y_test, y_pred_sel)

# ======================================
# (e) PERFORMANCE COMPARISON
# ======================================
print("\nModel Performance Comparison")
print("-----------------------------")
print("Accuracy with ALL features     :", acc_all)
print("Accuracy with SELECTED features:", acc_sel)
