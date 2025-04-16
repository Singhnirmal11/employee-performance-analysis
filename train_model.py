import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Simulate some training data
data = {
    'age': [25, 30, 45, 35, 50],
    'experience': [2, 5, 20, 10, 30],
    'projects': [3, 5, 10, 8, 12],
    'hours': [40, 45, 50, 60, 55],
    'training': [10, 20, 15, 25, 5],
    'satisfaction': [0.6, 0.8, 0.9, 0.7, 0.4],
    'performance': [70, 80, 95, 85, 65]
}

df = pd.DataFrame(data)

X = df.drop('performance', axis=1)
y = df['performance']

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, 'employee_performance_model.pkl')
print("Model saved as employee_performance_model.pkl")
