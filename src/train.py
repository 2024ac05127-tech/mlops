import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("data/heart.csv")

X = df.drop("num", axis=1)
y = df["num"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
print(f"Random Forest Accuracy: {rf_acc:.4f}")

# Select best model
if lr_acc > rf_acc:
    best_model = lr
    best_model_name = "Logistic Regression"
    best_acc = lr_acc
else:
    best_model = rf
    best_model_name = "Random Forest"
    best_acc = rf_acc

# Save best model
joblib.dump(best_model, "heart_disease_model.pkl")

# Save training log
with open("training_log.txt", "w") as f:
    f.write(f"Best Model: {best_model_name}\n")
    f.write(f"Accuracy: {best_acc:.4f}\n")
    f.write(f"Logistic Regression Accuracy: {lr_acc:.4f}\n")
    f.write(f"Random Forest Accuracy: {rf_acc:.4f}\n")

print(f"Best Model: {best_model_name}")
print("Model Saved")
