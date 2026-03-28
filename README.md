# Credit Card Fraud Detection using Machine Learning

## **Project Overview**

This project focuses on detecting fraudulent credit card transactions using Machine Learning.
The system analyzes transaction patterns and predicts whether a transaction is Fraud or Safe along with Risk Level.

This project simulates real-world fraud detection systems used by companies like
	•	Visa
	•	Mastercard
	•	PayPal


 ## Project Features
	•	Fraud detection using Machine Learning
	•	Model comparison (Logistic Regression, Random Forest, XGBoost)
	•	Imbalanced dataset handling
	•	Threshold tuning
	•	Risk level prediction
	•	Backend API using Flask
	•	Confusion matrix visualization
	•	Real-world fraud detection pipeline

## Dataset Used

**Dataset**: Credit Card Fraud Detection Dataset

**Dataset Details:** <br>
	•	284,807 transactions <br>
	•	492 fraud transactions<br>
	•	Highly imbalanced dataset

**Features:**<br>
	•	Time<br>
	•	V1–V28 (PCA features)<br>
	•	Amount<br>
	•	Class (Target)

**Class:**<br>
	•	0 → Safe<br>
	•	1 → Fraud

  **Technologies Used**<br>
	•	Python<br>
	•	Pandas<br>
	•	NumPy<br>
	•	Scikit-learn<br>
	•	XGBoost<br>
	•	Flask<br>
	•	Matplotlib<br>
	•	Joblib<br>

  ## Project Structure
  
  fraud-detection-project
│
├── train.py
├── app.py
├── fraud_model.pkl
├── scaler.pkl
├── threshold.pkl
├── sample_fraud.json
└── README.md

## Machine Learning Models Used

**Models Compared:**
	•	Logistic Regression
	•	Random Forest
	•	XGBoost

**Final Model Selected:**

XGBoost — Best performance

## Model Performance

**Final Results:**

Accuracy: 99.94%
Precision: 0.87
Recall: 0.82
F1 Score: 0.84

## Confusion Matrix:
	•	True Safe → 56852
	•	True Fraud → 80
	•	False Positive → 12
	•	False Negative → 18

## Risk Level Classification
Risk levels based on fraud probability:

Probability < 0.02
Risk Level = Low
Probability 0.02 - 0.1
Risk Level = MEDIUM
Probability> 0.1
Risk Level = HIGH
Example Output:
Prediction: Fraud
Probability: 0.99
Risk Level: HIGH

## API Usage

**Run backend server:**
python3 app.py

**Test API:**
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{"features":[values]}'

## Example Output

**Fraud Example**
{
 "prediction": "Fraud",
 "probability": 0.99,
 "risk_level": "HIGH"
}

**Safe Example**
{
 "prediction": "Safe",
 "probability": 0.00008,
 "risk_level": "LOW"
}

## Challenges Faced
	•	Imbalanced dataset
	•	PCA feature interpretation
	•	Threshold tuning
	•	Model selection
	•	Backend integration

## Why Custom Inputs Show Safe

Dataset uses PCA transformed features.

Because of this:
	•	Random values do not match fraud patterns
	•	Model predicts Safe

This is expected behavior in fraud detection systems.

## Future Improvements
	•	Frontend Dashboard
	•	Real-time fraud detection
	•	Explainable AI
	•	Cloud deployment
	•	Alert system
	•	Mobile integration

## Intended End Users
	•	Banks
	•	Payment gateways
	•	E-commerce platforms
	•	Fraud monitoring teams

## Conclusion

This project builds a Credit Card Fraud Detection System using Machine Learning.

The system:
	•	Detects fraud
	•	Provides probability
	•	Assigns risk level

This can be extended into a real-world fraud detection platform.

## Author

**Archana Kumar**
**Major Project — Credit Card Fraud Detection**
