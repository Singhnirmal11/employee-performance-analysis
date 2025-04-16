from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('employee_performance_model.pkl')
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect features from form
        age = float(request.form['age'])
        experience = float(request.form['experience'])
        projects = float(request.form['projects'])
        hours = float(request.form['hours'])
        training = float(request.form['training'])
        satisfaction = float(request.form['satisfaction'])

        features = np.array([[age, experience, projects, hours, training, satisfaction]])
        prediction = model.predict(features)

        return render_template('index.html', prediction_text=f"Predicted Performance Score: {prediction[0]:.2f}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
