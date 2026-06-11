import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.feature_selection import SelectFromModel

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# Create required folders
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Load cleaned dataset
df = pd.read_csv("cleaned_data/Cleaned_CreditScoring.csv")

print("\nDataset Loaded Successfully!")

# Remove duplicate rows
df = df.drop_duplicates()

# Remove invalid target values
df = df[df['Status'] != 0]

# Convert target labels
# 1 -> Good Credit -> 0
# 2 -> Bad Credit -> 1
df['Status'] = df['Status'].map({1: 0, 2: 1})

print("\nTarget Labels Cleaned Successfully!")

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Advanced Feature Engineering
df['Debt_Income_Ratio'] = df['Debt'] / (df['Income'] + 1)

df['Loan_Income_Ratio'] = df['Amount'] / (df['Income'] + 1)

df['Expense_Income_Ratio'] = df['Expenses'] / (df['Income'] + 1)

df['Debt_Asset_Ratio'] = df['Debt'] / (df['Assets'] + 1)

df['Savings_Estimate'] = df['Income'] - df['Expenses']

df['Net_Worth'] = df['Assets'] - df['Debt']

print("\nAdvanced Feature Engineering Completed!")

# Features and Target
X = df.drop('Status', axis=1)

# Handle categorical columns
X = pd.get_dummies(X, drop_first=True)

y = df['Status']

# Save original feature names
original_feature_names = X.columns.tolist()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain-Test Split Completed!")

# Apply Moderate SMOTE
smote = SMOTE(
    sampling_strategy=0.5,
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nSMOTE Applied Successfully!")
print("Balanced Training Dataset Created!")

# Base XGBoost Model
base_model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

# Hyperparameter Search Space
param_grid = {
    'n_estimators': [200, 300, 400],
    'learning_rate': [0.01, 0.03, 0.05],
    'max_depth': [4, 5, 6],
    'min_child_weight': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.5, 1],
    'reg_lambda': [1, 1.5, 2]
}

# Stratified Cross Validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Randomized Search
random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_grid,
    n_iter=20,
    scoring='roc_auc',
    cv=cv,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Train Hyperparameter Search
random_search.fit(
    X_train_smote,
    y_train_smote
)

print("\nHyperparameter Tuning Completed!")

# Best Model
xgb_model = random_search.best_estimator_

print("\nBest Parameters:")
print(random_search.best_params_)

# Initial Feature Selection Model
xgb_model.fit(
    X_train_smote,
    y_train_smote
)

# Feature Selection
selector = SelectFromModel(
    xgb_model,
    threshold='median',
    prefit=True
)

# Transform datasets
X_train_selected = selector.transform(X_train_smote)

X_test_selected = selector.transform(X_test)

# Get selected feature names
selected_features = np.array(
    original_feature_names
)[selector.get_support()]

# Final Model Training
xgb_model.fit(
    X_train_selected,
    y_train_smote
)

print("\nFinal Optimized XGBoost Model Trained Successfully!")

# Prediction Probabilities
y_prob = xgb_model.predict_proba(
    X_test_selected
)[:, 1]

# Balanced Threshold
threshold = 0.45

# Final Predictions
y_pred = (y_prob >= threshold).astype(int)

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_prob)

print("\nMODEL PERFORMANCE")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

# Classification Report
print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("outputs/confusion_matrix.png")

plt.show()

print("\nConfusion Matrix Saved!")

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Importance': xgb_model.feature_importances_
})

# Sort Feature Importance
feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print(feature_importance)

# Feature Importance Plot
plt.figure(figsize=(12, 7))

sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance
)

plt.title("Feature Importance")

plt.savefig("outputs/feature_importance.png")

plt.show()

print("\nFeature Importance Graph Saved!")

# Save Model
joblib.dump(
    xgb_model,
    "models/final_xgboost_model.pkl"
)

# Save Feature Selector
joblib.dump(
    selector,
    "models/feature_selector.pkl"
)

# Save Column Names
joblib.dump(
    original_feature_names,
    "models/model_columns.pkl"
)

print("\nFinal Optimized Model Saved Successfully!")

print("\nSaved Files:")
print("models/final_xgboost_model.pkl")
print("models/feature_selector.pkl")
print("models/model_columns.pkl")

print("\nCREDIT SCORING PROJECT COMPLETED!")