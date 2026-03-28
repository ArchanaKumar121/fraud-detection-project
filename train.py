import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# =========================
# Load Dataset
# =========================

df = pd.read_csv("../data/creditcard.csv")

# Split X and y
X = df.drop("Class", axis=1)
y = df["Class"]


# =========================
# Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# Scaling Amount Column
# =========================

scaler = StandardScaler()

X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]])
X_test["Amount"] = scaler.transform(X_test[["Amount"]])

print("Scaling done")


# =========================
# Logistic Regression
# =========================

print("\n===== Logistic Regression =====")

lr = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

lr.fit(X_train, y_train)

y_prob_lr = lr.predict_proba(X_test)[:, 1]
y_pred_lr = (y_prob_lr > 0.05).astype(int)

print(classification_report(y_test, y_pred_lr))


# =========================
# Random Forest
# =========================

print("\n===== Random Forest =====")

rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print(classification_report(y_test, y_pred_rf))


# =========================
# XGBoost (Best Model)
# =========================

print("\n===== XGBoost =====")

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=100,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

xgb.fit(X_train, y_train)

y_prob = xgb.predict_proba(X_test)[:, 1]


# =========================
# Threshold Tuning
# =========================

print("\n===== Threshold Tuning =====")

thresholds = [0.01, 0.03, 0.05, 0.1]

best_threshold = 0.05

for threshold in thresholds:
    print(f"\nThreshold: {threshold}")

    y_pred = (y_prob > threshold).astype(int)

    print(classification_report(y_test, y_pred))


# Final prediction using best threshold
y_pred_xgb = (y_prob > best_threshold).astype(int)

print("\n===== Final XGBoost Model =====")

print(classification_report(y_test, y_pred_xgb))


# =========================
# Save Model
# =========================

os.makedirs("../models", exist_ok=True)

joblib.dump(xgb, "../models/fraud_model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")
joblib.dump(best_threshold, "../models/threshold.pkl")

print("\n✅ Model, scaler and threshold saved!")

# Save one fraud example
fraud_sample = X_test[y_test == 1].iloc[0]

import json

with open("../models/sample_fraud.json", "w") as f:
    json.dump(fraud_sample.tolist(), f)

print("✅ Fraud sample saved")

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Predict
y_pred = xgb.predict(X_test)

print("\nModel Performance")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy")
print(accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

plt.title("Confusion Matrix - XGBoost")
plt.show()

