import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report

# Load Ames Housing dataset (local CSV)
data = pd.read_csv("AmesHousing.csv")

# Select numeric features only
data_numeric = data.select_dtypes(include=[np.number])

X = data_numeric.drop("SalePrice", axis=1)
y = data_numeric["SalePrice"]

# ---------------- Linear Regression ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred = lin_model.predict(X_test)

print("Linear Regression Results")
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# ---------------- Ridge Regression ----------------
ridge = Ridge()
params_ridge = {'alpha': [0.01, 0.1, 1, 10, 100]}

grid_ridge = GridSearchCV(ridge, params_ridge, cv=5)
grid_ridge.fit(X_train, y_train)

print("Best Ridge Alpha:", grid_ridge.best_params_)

# ---------------- Logistic Regression ----------------
# Convert prices into high vs low price houses
median_price = np.median(y)
y_binary = (y >= median_price).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42
)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=500)
log_model.fit(X_train, y_train)
y_pred = log_model.predict(X_test)

print("\nLogistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

params_log = {'C': [0.01, 0.1, 1, 10]}
grid_log = GridSearchCV(LogisticRegression(max_iter=500), params_log, cv=5)
grid_log.fit(X_train, y_train)

print("Best Logistic Parameters:", grid_log.best_params_)
