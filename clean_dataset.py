import pandas as pd
import numpy as np
import os

# Create cleaned_data folder
os.makedirs("cleaned_data", exist_ok=True)

# Load Dataset
df = pd.read_csv("data/CreditScoring.csv")

print("\nDataset Loaded Successfully!")

# Remove duplicate rows
df = df.drop_duplicates()

print("\nDuplicate Rows Removed!")

# Remove invalid target values
df = df[df['Status'] != 0]

print("\nInvalid Target Rows Removed!")

# Remove impossible financial rows
df = df[df['Income'] > 0]

df = df[df['Assets'] >= 0]

df = df[df['Debt'] >= 0]

df = df[df['Expenses'] >= 0]

df = df[df['Amount'] >= 0]

df = df[df['Price'] >= 0]

print("\nImpossible Financial Rows Removed!")

# Remove extreme outliers using IQR
numeric_columns = [
    'Income',
    'Debt',
    'Assets',
    'Amount',
    'Expenses',
    'Price'
]

for col in numeric_columns:

    Q1 = df[col].quantile(0.25)

    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - (1.5 * IQR)

    upper_bound = Q3 + (1.5 * IQR)

    df = df[
        (df[col] >= lower_bound) &
        (df[col] <= upper_bound)
    ]

print("\nOutliers Removed Successfully!")

# Apply Log Transformation
df['Income'] = np.log1p(df['Income'])

df['Assets'] = np.log1p(df['Assets'])

df['Debt'] = np.log1p(df['Debt'])

df['Amount'] = np.log1p(df['Amount'])

df['Expenses'] = np.log1p(df['Expenses'])

df['Price'] = np.log1p(df['Price'])

print("\nLog Transformation Applied Successfully!")

# Reset Index
df = df.reset_index(drop=True)

# Save Cleaned Dataset
cleaned_file_path = "cleaned_data/Cleaned_CreditScoring.csv"

df.to_csv(cleaned_file_path, index=False)

print("\nCleaned Dataset Saved Successfully!")

print(f"\nSaved File: {cleaned_file_path}")

# Dataset Information
print("\nFinal Dataset Shape:")
print(df.shape)

print("\nDataset Cleaning Completed!")