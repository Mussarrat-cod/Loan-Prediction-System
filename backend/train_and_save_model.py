"""
Train and Save Loan Prediction Model
=====================================
This script trains the Random Forest model with feature engineering
and saves it for use in the Flask API.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("TRAINING AND SAVING LOAN PREDICTION MODEL")
print("="*60)

# Load dataset
print("\n[1/5] Loading dataset...")
df = pd.read_csv("../loan_data.csv")
print(f"✓ Loaded {len(df)} records")

# Prepare data
df_original = df.copy()
df = df.drop('Loan_ID', axis=1)

# Create label encoders
print("\n[2/5] Creating label encoders...")
encoders = {}
categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])

print(f"✓ Created encoders for {len(encoders)} columns")

# Feature Engineering
print("\n[3/5] Engineering features...")
df_original_clean = df_original.drop('Loan_ID', axis=1)

# Create new features
df_features = df_original_clean.copy()
df_features['Total_Income'] = df_features['ApplicantIncome'] + df_features['CoapplicantIncome']
df_features['Income_Loan_Ratio'] = df_features['Total_Income'] / (df_features['LoanAmount'] + 1)
df_features['LoanAmount_log'] = np.log(df_features['LoanAmount'] + 1)
df_features['Total_Income_log'] = np.log(df_features['Total_Income'] + 1)

# Encode categorical variables
for col in df_features.select_dtypes(include='object').columns:
    df_features[col] = encoders[col].transform(df_features[col])

print("✓ Created 4 engineered features")

# Split data
X = df_features.drop("Loan_Status", axis=1)
y = df_features["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
print("\n[4/5] Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✓ Model trained successfully!")
print(f"  Accuracy: {accuracy*100:.2f}%")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 Important Features:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Save model and encoders
print("\n[5/5] Saving model and encoders...")
joblib.dump(model, 'model.pkl')
joblib.dump(encoders, 'encoders.pkl')
joblib.dump(list(X.columns), 'feature_names.pkl')

# Save model metadata
model_info = {
    'accuracy': float(accuracy),
    'n_samples_train': len(X_train),
    'n_samples_test': len(X_test),
    'n_features': len(X.columns),
    'feature_importance': feature_importance.head(10).to_dict('records'),
    'model_type': 'RandomForestClassifier',
    'n_estimators': 100
}

with open('model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("✓ Saved model.pkl")
print("✓ Saved encoders.pkl")
print("✓ Saved feature_names.pkl")
print("✓ Saved model_info.json")

print("\n" + "="*60)
print("MODEL TRAINING COMPLETE!")
print("="*60)
print(f"\nModel Accuracy: {accuracy*100:.2f}%")
print("Ready to use in Flask API!")
