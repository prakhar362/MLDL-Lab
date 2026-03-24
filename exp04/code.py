import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


np.random.seed(42)
data_size = 300
data = {
    "study_hours": np.random.normal(5, 2, data_size),
    "attendance": np.random.normal(75, 10, data_size),
    "internal_score": np.random.normal(60, 15, data_size),
    "sleep_hours": np.random.normal(7, 1.5, data_size),
    "previous_gpa": np.random.normal(6.5, 1, data_size)
}


df = pd.DataFrame(data)
# Create target variable
df["result"] = (
    (df["study_hours"] > 4) &
    (df["attendance"] > 70) &
    (df["internal_score"] > 55) &
    (df["previous_gpa"] > 6)
).astype(int)
df.head()
X = df.drop("result", axis=1)
y = df["result"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


k = 5  # odd value to avoid tie
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
k_values = range(1, 21, 2)
accuracy_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
    accuracy_scores.append(scores.mean())
plt.figure(figsize=(8,5))
plt.plot(k_values, accuracy_scores, marker='o')
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Cross-Validated Accuracy")
plt.title("KNN: Accuracy vs k")
plt.grid(True)
plt.show()
optimal_k = k_values[np.argmax(accuracy_scores)]
print("Optimal k:", optimal_k)
