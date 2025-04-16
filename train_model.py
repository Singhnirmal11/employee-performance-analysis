from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd

# Example: Load sample data (replace with your actual data)
data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6], 'target': [0, 1, 0]})
X_train = data[['feature1', 'feature2']]
y_train = data['target']

# Train and save the model
model = RandomForestClassifier()
model.fit(X_train, y_train)
joblib.dump(model, 'employee_performance_model.pkl')
print("Model saved successfully.")