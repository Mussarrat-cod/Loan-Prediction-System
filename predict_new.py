"""
Interactive Loan Approval Predictor
====================================
Use this script to make predictions on new loan applications.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("INTERACTIVE LOAN APPROVAL PREDICTOR")
print("="*60)

# Load and prepare training data
df = pd.read_csv("loan_data.csv")
df = df.drop('Loan_ID', axis=1)

# Encode categorical variables
le_dict = {}
for col in df.select_dtypes(include='object').columns:
    le_dict[col] = LabelEncoder()
    df[col] = le_dict[col].fit_transform(df[col])

# Feature engineering
df_enhanced = pd.read_csv("loan_data.csv")
df_enhanced = df_enhanced.drop('Loan_ID', axis=1)

df_enhanced['Total_Income'] = df_enhanced['ApplicantIncome'] + df_enhanced['CoapplicantIncome']
df_enhanced['Income_Loan_Ratio'] = df_enhanced['Total_Income'] / (df_enhanced['LoanAmount'] + 1)
df_enhanced['LoanAmount_log'] = np.log(df_enhanced['LoanAmount'] + 1)
df_enhanced['Total_Income_log'] = np.log(df_enhanced['Total_Income'] + 1)

for col in df_enhanced.select_dtypes(include='object').columns:
    df_enhanced[col] = le_dict[col].fit_transform(df_enhanced[col])

# Train the best model
X = df_enhanced.drop("Loan_Status", axis=1)
y = df_enhanced["Loan_Status"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

print("\n[INFO] Model trained successfully!")
print("[INFO] Model: Random Forest with Feature Engineering")

# Example predictions
print("\n" + "="*60)
print("EXAMPLE PREDICTIONS")
print("="*60)

# Example 1: Good candidate
example1 = pd.DataFrame({
    'Gender': ['Male'],
    'Married': ['Yes'],
    'Dependents': ['2'],
    'Education': ['Graduate'],
    'Self_Employed': ['No'],
    'ApplicantIncome': [5000],
    'CoapplicantIncome': [2000],
    'LoanAmount': [150],
    'Loan_Amount_Term': [360],
    'Credit_History': [1],
    'Property_Area': ['Urban']
})

# Feature engineering for example
example1['Total_Income'] = example1['ApplicantIncome'] + example1['CoapplicantIncome']
example1['Income_Loan_Ratio'] = example1['Total_Income'] / (example1['LoanAmount'] + 1)
example1['LoanAmount_log'] = np.log(example1['LoanAmount'] + 1)
example1['Total_Income_log'] = np.log(example1['Total_Income'] + 1)

# Encode
for col in example1.select_dtypes(include='object').columns:
    example1[col] = le_dict[col].transform(example1[col])

prediction = model.predict(example1)[0]
probability = model.predict_proba(example1)[0]

print("\nExample 1: High-income Graduate with Good Credit")
print("  Applicant Income: $5,000")
print("  Co-applicant Income: $2,000")
print("  Loan Amount: $150,000")
print("  Credit History: Good")
print(f"  Prediction: {'APPROVED' if prediction == 1 else 'REJECTED'}")
print(f"  Confidence: {max(probability)*100:.2f}%")

# Example 2: Risky candidate
example2 = pd.DataFrame({
    'Gender': ['Male'],
    'Married': ['No'],
    'Dependents': ['0'],
    'Education': ['Not Graduate'],
    'Self_Employed': ['Yes'],
    'ApplicantIncome': [2000],
    'CoapplicantIncome': [0],
    'LoanAmount': [200],
    'Loan_Amount_Term': [360],
    'Credit_History': [0],
    'Property_Area': ['Rural']
})

# Feature engineering
example2['Total_Income'] = example2['ApplicantIncome'] + example2['CoapplicantIncome']
example2['Income_Loan_Ratio'] = example2['Total_Income'] / (example2['LoanAmount'] + 1)
example2['LoanAmount_log'] = np.log(example2['LoanAmount'] + 1)
example2['Total_Income_log'] = np.log(example2['Total_Income'] + 1)

# Encode
for col in example2.select_dtypes(include='object').columns:
    example2[col] = le_dict[col].transform(example2[col])

prediction2 = model.predict(example2)[0]
probability2 = model.predict_proba(example2)[0]

print("\nExample 2: Low-income Non-graduate with Bad Credit")
print("  Applicant Income: $2,000")
print("  Co-applicant Income: $0")
print("  Loan Amount: $200,000")
print("  Credit History: Poor")
print(f"  Prediction: {'APPROVED' if prediction2 == 1 else 'REJECTED'}")
print(f"  Confidence: {max(probability2)*100:.2f}%")

print("\n" + "="*60)
print("You can modify this script to make your own predictions!")
print("="*60)
