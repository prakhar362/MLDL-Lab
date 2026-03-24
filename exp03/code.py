
# ===============================
# 1. Import Libraries
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===============================
# 2. Load Dataset
# ===============================
url = "https://www.kaggle.com/code/cristianlapenta/wine-dataset-sklearn-machine-learning-project"

columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

df = pd.read_csv(url, names=columns)

print("Dataset Preview:")
print(df.head())

# ===============================
# 3. Data Preprocessing
# ===============================

# Encode all categorical columns
le = LabelEncoder()

for col in df.columns:
    df[col] = le.fit_transform(df[col])

# Split features and target
X = df.drop('class', axis=1)
y = df['class']

# ===============================
# 4. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 5. Decision Tree Model
# ===============================
dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

# ===============================
# 6. Random Forest Model
# ===============================
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

# ===============================
# 7. Evaluation Function
# ===============================
def evaluate_model(name, y_test, y_pred):
    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ===============================
# 8. Evaluate Models
# ===============================
evaluate_model("Decision Tree", y_test, y_pred_dt)
evaluate_model("Random Forest", y_test, y_pred_rf)

# ===============================
# 9. Confusion Matrix (Separate)
# ===============================

# Decision Tree
plt.figure()
sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d')
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Random Forest
plt.figure()
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ===============================
# 10. Decision Tree Visualization
# ===============================

from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.tree import plot_tree


fig, axes = plt.subplots(1, 2, figsize=(16, 6))
cm_dt = confusion_matrix(y_test, dt_pred)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=data.target_names, yticklabels=data.target_names)
axes[0].set_title('Decision Tree: Performance', fontsize=15, fontweight='bold')


cm_rf = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=data.target_names, yticklabels=data.target_names)
axes[1].set_title('Random Forest: Performance', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()


# --- PLOT 2: Performance Metrics Comparison ---
plt.figure(figsize=(10, 6))
metrics = {
    'Accuracy': [accuracy_score(y_test, dt_pred), accuracy_score(y_test, rf_pred)],
    'Precision': [precision_score(y_test, dt_pred, average='macro'), precision_score(y_test, rf_pred, average='macro')],
    'Recall': [recall_score(y_test, dt_pred, average='macro'), recall_score(y_test, rf_pred, average='macro')]
}
x = np.arange(len(metrics))
width = 0.35
plt.bar(x - width/2, [m[0] for m in metrics.values()], width, label='Decision Tree', color='#5DADE2', edgecolor='black')
plt.bar(x + width/2, [m[1] for m in metrics.values()], width, label='Random Forest', color='#E74C3C', edgecolor='black')
plt.xticks(x, metrics.keys())
plt.ylabel('Score (0.0 - 1.0)')
plt.title('Model Metric Comparison', fontsize=16, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.show()


# --- PLOT 3: Feature Importance (What the AI looked at) ---
plt.figure(figsize=(10, 7))
importances = rf_model.feature_importances_
indices = np.argsort(importances)
plt.title('Feature Importances: Wine Classification', fontsize=16, fontweight='bold')
plt.barh(range(len(indices)), importances[indices], color='#16A085', align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# --- PLOT 4: Visualize the Tree Logic ---
plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=data.feature_names, class_names=data.target_names,
          filled=True, rounded=True, fontsize=12)
plt.title("Decision Tree Logic Flowchart", fontsize=20, fontweight='bold')
plt.show()

