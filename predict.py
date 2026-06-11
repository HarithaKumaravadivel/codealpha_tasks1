import joblib
import pandas as pd

model = joblib.load("models/random_forest_model.pkl")

print("Model Loaded Successfully!")

sample_data = pd.DataFrame([{
    'Seniority': 5,
    'Home': 1,
    'Time': 12,
    'Age': 35,
    'Marital': 1,
    'Records': 0,
    'Job': 1,
    'Expenses': 2000,
    'Income': 5000,
    'Assets': 10000,
    'Debt': 1000,
    'Amount': 3000,
    'Price': 4000,
    'Debt_Income_Ratio': 1000 / (5000 + 1)
}])

prediction = model.predict(sample_data)

if prediction[0] == 0:
    print("Good Credit Risk")
else:
    print("Bad Credit Risk")