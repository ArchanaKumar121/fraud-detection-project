from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model & scaler
model = joblib.load("../models/fraud_model.pkl")
scaler = joblib.load("../models/scaler.pkl")
threshold = joblib.load("../models/threshold.pkl")

@app.route("/")
def home():
    return "Fraud Detection API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Example input: list of 30 values
    features = np.array(data["features"]).reshape(1, -1)

    # Scale Amount
    features[0][29] = scaler.transform([[features[0][29]]])[0][0]

    # Predict probability
    prob = model.predict_proba(features)[0][1]

    # Load threshold
    threshold = joblib.load("../models/threshold.pkl")

    # Prediction
    if prob > threshold:
        result = "Fraud"
    else:
        result = "Safe"

    # Risk level
    if prob > 0.8:
        risk = "HIGH"
    elif prob > 0.01:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return jsonify({
        "prediction": result,
        "probability": float(prob),
        "risk_level": risk
    })
if __name__ == "__main__":
    app.run(debug=True)