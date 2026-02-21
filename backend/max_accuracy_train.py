"""
Maximum Accuracy Loan Prediction Model
======================================
Focus on achieving highest possible accuracy with ensemble methods
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("MAXIMUM ACCURACY LOAN PREDICTION MODEL")
print("="*70)

# Load and prepare data
print("\n[1/6] Loading and preparing data...")
train_df = pd.read_csv("../train.csv")

df = train_df.copy()

# Clean data
df['Dependents'] = df['Dependents'].replace('3+', '3')
df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce').fillna(0)

# Handle missing values
for col in df.columns:
    if col == 'Loan_Status':
        continue
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

print(f"✓ Cleaned {len(df)} records")

# Smart feature engineering
print("\n[2/6] Smart feature engineering...")
df_features = df.copy()

# Key features that matter most
df_features['Total_Income'] = df_features['ApplicantIncome'] + df_features['CoapplicantIncome']
df_features['Loan_to_Income_Ratio'] = df_features['LoanAmount'] / (df_features['Total_Income'] + 1)
df_features['Credit_History_x_Income'] = df_features['Credit_History'] * df_features['Total_Income']
df_features['Good_Credit_Flag'] = (df_features['Credit_History'] == 1).astype(int)

# Log transforms for skewed data
df_features['LoanAmount_log'] = np.log1p(df_features['LoanAmount'])
df_features['Total_Income_log'] = np.log1p(df_features['Total_Income'])

print(f"✓ Created 5 key features")

# Encode categorical variables
print("\n[3/6] Encoding variables...")
encoders = {}
categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']

for col in categorical_cols:
    encoders[col] = LabelEncoder()
    df_features[col] = encoders[col].fit_transform(df_features[col])

# Encode target
df_features['Loan_Status'] = df_features['Loan_Status'].map({'Y': 1, 'N': 0})

print(f"✓ Encoded {len(categorical_cols)} categorical variables")

# Select most important features
print("\n[4/6] Feature selection...")
feature_cols = [
    'Credit_History', 'Good_Credit_Flag', 'Credit_History_x_Income',
    'Loan_to_Income_Ratio', 'Total_Income_log', 'LoanAmount_log',
    'Married', 'Education', 'Property_Area', 'ApplicantIncome',
    'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'
]

X = df_features[feature_cols]
y = df_features['Loan_Status']

print(f"✓ Selected {len(feature_cols)} key features")

# Create powerful ensemble
print("\n[5/6] Building maximum accuracy ensemble...")

# Individual optimized models
rf1 = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
rf2 = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=100)
rf3 = RandomForestClassifier(n_estimators=250, max_depth=18, random_state=200)

dt = DecisionTreeClassifier(max_depth=10, random_state=42)
lr = LogisticRegression(C=1.0, random_state=42, max_iter=1000)

# Create voting ensemble
ensemble = VotingClassifier(
    estimators=[
        ('rf1', rf1),
        ('rf2', rf2), 
        ('rf3', rf3),
        ('dt', dt),
        ('lr', lr)
    ],
    voting='soft'  # Use probability voting
)

# Test individual models and ensemble
models = {
    'Random Forest 1': rf1,
    'Random Forest 2': rf2,
    'Random Forest 3': rf3,
    'Decision Tree': dt,
    'Logistic Regression': lr,
    'Ensemble': ensemble
}

best_model = None
best_accuracy = 0
best_name = ""

print("\nModel comparison:")
for name, model in models.items():
    # Use stratified cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    
    print(f"{name:20}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
    
    if scores.mean() > best_accuracy:
        best_accuracy = scores.mean()
        best_model = model
        best_name = name

print(f"\n🏆 Best model: {best_name} with {best_accuracy*100:.2f}% accuracy")

# Final training and validation
print("\n[6/6] Final training and validation...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train best model on full training set
best_model.fit(X_train, y_train)

# Final predictions
y_pred = best_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)

print(f"Final test accuracy: {final_accuracy*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the best model
print("\n" + "="*70)
print("SAVING MAXIMUM ACCURACY MODEL")
print("="*70)

joblib.dump(best_model, 'model.pkl')
joblib.dump(encoders, 'encoders.pkl')
joblib.dump(feature_cols, 'feature_names.pkl')

# Save model info
model_info = {
    'best_model': best_name,
    'cv_accuracy': float(best_accuracy),
    'test_accuracy': float(final_accuracy),
    'previous_accuracy': 0.8495,
    'improvement': float(final_accuracy - 0.8495),
    'n_features': len(feature_cols),
    'n_samples': len(X),
    'features': feature_cols,
    'model_type': type(best_model).__name__
}

if hasattr(best_model, 'feature_importances_'):
    importance = list(zip(feature_cols, best_model.feature_importances_))
    importance.sort(key=lambda x: x[1], reverse=True)
    model_info['feature_importance'] = [{'feature': f, 'importance': float(i)} for f, i in importance]

with open('model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("✓ Saved maximum accuracy model.pkl")
print("✓ Saved encoders.pkl")
print("✓ Saved feature_names.pkl")
print("✓ Saved model_info.json")

print("\n" + "="*70)
print("MAXIMUM ACCURACY MODEL COMPLETE!")
print("="*70)
print(f"\n🎯 Cross-Validation Accuracy: {best_accuracy*100:.2f}%")
print(f"🎯 Test Accuracy: {final_accuracy*100:.2f}%")
print(f"📈 Improvement: {(final_accuracy - 0.8495)*100:+.2f}%")
print(f"🏆 Best Model: {best_name}")
print(f"🧠 Features: {len(feature_cols)}")
print("\n🚀 MAXIMUM ACCURACY MODEL READY!")
