import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create dataset
np.random.seed(42)

data = {
    "Age": np.random.randint(30, 80, 100),
    "BloodPressure": np.random.randint(90, 180, 100),
    "Cholesterol": np.random.randint(150, 320, 100),
    "Target": np.random.randint(0, 2, 100)
}

df = pd.DataFrame(data)

print("First 5 rows of dataset:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

# Features and target
X = df[["Age", "BloodPressure", "Cholesterol"]]
y = df["Target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# SVM
svm = SVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

# Accuracy
print("\n==============================")
print("RESULT")
print("==============================")
print("Logistic Regression Accuracy:",
      accuracy_score(y_test, lr_pred))

print("SVM Accuracy:",
      accuracy_score(y_test, svm_pred))

print("\nTask Completed Successfully!")
