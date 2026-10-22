# 1. Import thu vien
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 2. Tai du lieu Iris
iris = load_iris()
X = iris.data
y = iris.target

print("So mau du lieu:", X.shape[0])
print("So dac trung:", X.shape[1])
print("Cac lop:", iris.target_names)

# 3. Chia du lieu: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Khoi tao hai mo hinh
dt = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42
)

# 5. Huan luyen
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# 6. Du doan
pred_dt = dt.predict(X_test)
pred_rf = rf.predict(X_test)

# 7. Danh gia ket qua
print("\n=== DECISION TREE ===")
print("Accuracy:", accuracy_score(y_test, pred_dt))
print(classification_report(
    y_test, pred_dt,
    target_names=iris.target_names,
    zero_division=0
))

print("\n=== RANDOM FOREST ===")
print("Accuracy:", accuracy_score(y_test, pred_rf))
print(classification_report(
    y_test, pred_rf,
    target_names=iris.target_names,
    zero_division=0
))

# 8. Ve cay quyet dinh
plt.figure(figsize=(16, 8))
plot_tree(
    dt,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True
)
plt.title("Decision Tree - Iris")
plt.tight_layout()
plt.show()

# 9. Ve confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test, pred_dt,
    display_labels=iris.target_names,
    ax=axes[0]
)
axes[0].set_title("Decision Tree")

ConfusionMatrixDisplay.from_predictions(
    y_test, pred_rf,
    display_labels=iris.target_names,
    ax=axes[1]
)
axes[1].set_title("Random Forest")

plt.tight_layout()
plt.show()

# 10. So sanh do chinh xac
scores = [
    accuracy_score(y_test, pred_dt),
    accuracy_score(y_test, pred_rf)
]

plt.figure(figsize=(7, 4))
plt.bar(["Decision Tree", "Random Forest"], scores)
plt.ylim(0, 1.05)
plt.ylabel("Accuracy")
plt.title("So sanh hai thuat toan")
plt.show()
