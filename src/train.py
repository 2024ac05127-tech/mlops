import pandas as pd
import joblib
from sklearn.impute import SimpleImputer

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
# Load data
df = pd.read_csv("data/heart.csv")

X = df.drop("num", axis=1)
y = df["num"]
imputer = SimpleImputer(strategy="median")

X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==================================================
# LOGISTIC REGRESSION
# ==================================================

lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

lr_acc = accuracy_score(
    y_test,
    lr_pred
)

lr_precision = precision_score(
    y_test,
    lr_pred,
    average="weighted"
)

lr_recall = recall_score(
    y_test,
    lr_pred,
    average="weighted"
)

lr_f1 = f1_score(
    y_test,
    lr_pred,
    average="weighted"
)

if len(y.unique()) == 2:
    lr_auc = roc_auc_score(
        y_test,
        lr.predict_proba(X_test)[:, 1]
    )
else:
    lr_auc = 0

print("\nLogistic Regression Results")
print(classification_report(y_test, lr_pred))

# ==================================================
# RANDOM FOREST
# ==================================================

rf = RandomForestClassifier(
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_acc = accuracy_score(
    y_test,
    rf_pred
)

rf_precision = precision_score(
    y_test,
    rf_pred,
    average="weighted"
)

rf_recall = recall_score(
    y_test,
    rf_pred,
    average="weighted"
)

rf_f1 = f1_score(
    y_test,
    rf_pred,
    average="weighted"
)

if len(y.unique()) == 2:
    rf_auc = roc_auc_score(
        y_test,
        rf.predict_proba(X_test)[:, 1]
    )
else:
    rf_auc = 0

print("\nRandom Forest Results")
print(classification_report(y_test, rf_pred))

# ==================================================
# MLFLOW EXPERIMENT TRACKING
# ==================================================

mlflow.set_experiment(
    "Heart_Disease_Classification"
)

# -----------------------------
# Logistic Regression Run
# -----------------------------

with mlflow.start_run(
    run_name="LogisticRegression"
):

    mlflow.log_param(
        "model_name",
        "LogisticRegression"
    )

    mlflow.log_param(
        "max_iter",
        1000
    )

    mlflow.log_metric(
        "accuracy",
        lr_acc
    )

    mlflow.log_metric(
        "precision",
        lr_precision
    )

    mlflow.log_metric(
        "recall",
        lr_recall
    )

    mlflow.log_metric(
        "f1_score",
        lr_f1
    )

    mlflow.log_metric(
        "roc_auc",
        lr_auc
    )

    mlflow.sklearn.log_model(
        sk_model=lr,
        name="LogisticRegression_Model"
    )

# -----------------------------
# Random Forest Run
# -----------------------------

with mlflow.start_run(
    run_name="RandomForest"
):

    mlflow.log_param(
        "model_name",
        "RandomForest"
    )

    mlflow.log_param(
        "n_estimators",
        rf.n_estimators
    )

    mlflow.log_metric(
        "accuracy",
        rf_acc
    )

    mlflow.log_metric(
        "precision",
        rf_precision
    )

    mlflow.log_metric(
        "recall",
        rf_recall
    )

    mlflow.log_metric(
        "f1_score",
        rf_f1
    )

    mlflow.log_metric(
        "roc_auc",
        rf_auc
    )

    mlflow.sklearn.log_model(
        sk_model=rf,
        name="RandomForest_Model"
    )

# ==================================================
# SELECT BEST MODEL
# ==================================================

if lr_acc > rf_acc:

    best_model = lr
    best_model_name = "LogisticRegression"
    best_acc = lr_acc

else:

    best_model = rf
    best_model_name = "RandomForest"
    best_acc = rf_acc

# ==================================================
# SAVE BEST MODEL
# ==================================================

joblib.dump(
    best_model,
    "heart_disease_model.pkl"
)

# Save imputer also
joblib.dump(
    imputer,
    "imputer.pkl"
)

# Training log

with open(
    "training_log.txt",
    "w"
) as f:

    f.write(
        f"Best Model: {best_model_name}\n"
    )

    f.write(
        f"Best Accuracy: {best_acc:.4f}\n"
    )

    f.write(
        f"Logistic Regression Accuracy: {lr_acc:.4f}\n"
    )

    f.write(
        f"Random Forest Accuracy: {rf_acc:.4f}\n"
    )

# ==================================================
# LOG BEST MODEL RUN
# ==================================================

with mlflow.start_run(
    run_name="Best_Model"
):

    mlflow.log_param(
        "best_model",
        best_model_name
    )

    mlflow.log_metric(
        "best_accuracy",
        best_acc
    )

    mlflow.log_artifact(
        "heart_disease_model.pkl"
    )

    mlflow.log_artifact(
        "training_log.txt"
    )

    mlflow.sklearn.log_model(
        sk_model=best_model,
        name="Best_Model"
    )

print("\n================================")
print("Training Complete")
print("================================")
print("Best Model:", best_model_name)
print("Best Accuracy:", round(best_acc, 4))
print("Model Saved: heart_disease_model.pkl")
print("MLflow Logging Completed")