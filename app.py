from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('employee_performance_model.pkl')

# Function to predict employee performance based on input features
def predict_performance(features):
    feature_names = ['age', 'experience']  # Replace with actual feature names
    features_df = pd.DataFrame([features], columns=feature_names)
    prediction = model.predict(features_df)
    return prediction[0]

# Default route
@app.route('/')
def home():
    return "Welcome to Employee Performance Analysis! Use /predict endpoint for predictions."

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = data['features']
    result = predict_performance(features)
    return jsonify({'prediction': int(result)})

if __name__ == "__main__":
    app.run(debug=True)