# ============================================

# SVM Classification with Hyperparameter Tuning # Dataset: Titanic (Kaggle)
# ============================================
# Step 1: Import Libraries import numpy as np
import pandas as pd

import matplotlib.pyplot as plt import seaborn as sns


from sklearn.model_selection import train_test_split, GridSearchCV from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



# Step 2: Load Dataset (Correct Path)

data = pd.read_csv('sample_data/train.csv')



# Step 3: Basic Data Cleaning



# Fill missing Age with median

data['Age'] = data['Age'].fillna(data['Age'].median())



# Fill missing Embarked with mode

data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])



# Drop unnecessary columns

data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)



# Convert categorical variables

data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})



# One-hot encoding for Embarked

data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)



# Step 4: Define Features and Target

X = data.drop('Survived', axis=1) y = data['Survived']


# Step 5: Train-Test Split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42
)



# Step 6: Feature Scaling (Very Important for SVM) scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) X_test = scaler.transform(X_test)


# Step 7: Baseline SVM Model svm_model = SVC() svm_model.fit(X_train, y_train)


y_pred = svm_model.predict(X_test)

print("Baseline Accuracy:", accuracy_score(y_test, y_pred))



# Step 8: Hyperparameter Tuning using GridSearchCV param_grid = {
'C': [0.1, 1, 10, 100],

'gamma': ['scale', 'auto'],

'kernel': ['linear', 'rbf']

}

grid = GridSearchCV(SVC(), param_grid, cv=5, verbose=1) grid.fit(X_train, y_train)


print("\nBest Parameters Found:", grid.best_params_)



# Step 9: Evaluate Tuned Model best_model = grid.best_estimator_ y_pred_best = best_model.predict(X_test)


print("\nTuned Model Accuracy:", accuracy_score(y_test, y_pred_best))



print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_best))



print("\nClassification Report:\n", classification_report(y_test, y_pred_best))
