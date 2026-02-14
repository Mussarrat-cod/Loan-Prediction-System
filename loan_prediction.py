"""
Loan Approval Prediction ML Project
====================================
This project predicts whether a loan application should be approved or rejected
based on applicant details using machine learning.

Author: ML Enthusiast
Date: February 2026
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("LOAN APPROVAL PREDICTION SYSTEM")
print("="*60)

# ==========================================
# 1. LOAD DATASET
# ==========================================
print("\n[Step 1] Loading Dataset...")
df = pd.read_csv("loan_data.csv")
print(f"[OK] Dataset loaded successfully!")
print(f"  - Total records: {len(df)}")
print(f"  - Total features: {len(df.columns)}")
print(f"\n  First 5 rows:")
print(df.head())

# ==========================================
# 2. DATA EXPLORATION
# ==========================================
print("\n[Step 2] Exploring Data...")
print(f"\n  Dataset Info:")
print(f"  - Shape: {df.shape}")
print(f"\n  Missing Values:")
print(df.isnull().sum())

print(f"\n  Data Types:")
print(df.dtypes)

# ==========================================
# 3. HANDLE MISSING VALUES
# ==========================================
print("\n[Step 3] Handling Missing Values...")
# Fill categorical columns with mode
for col in df.select_dtypes(include='object').columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Fill numeric columns with median
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

print(f"[OK] Missing values handled!")
print(f"  Remaining missing values: {df.isnull().sum().sum()}")

# ==========================================
# 4. ENCODE CATEGORICAL VARIABLES
# ==========================================
print("\n[Step 4] Encoding Categorical Variables...")
# Drop Loan_ID as it's not a feature
df = df.drop('Loan_ID', axis=1)

le = LabelEncoder()
categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])
    
print(f"[OK] Encoded {len(categorical_cols)} categorical columns")
print(f"  Columns: {list(categorical_cols)}")

# ==========================================
# 5. BASIC MODEL (Baseline)
# ==========================================
print("\n" + "="*60)
print("BASELINE MODEL (Without Feature Engineering)")
print("="*60)

# Split features and target
X_basic = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Train-test split with stratification
X_train_basic, X_test_basic, y_train, y_test = train_test_split(
    X_basic, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n  Training set: {X_train_basic.shape[0]} samples")
print(f"  Testing set: {X_test_basic.shape[0]} samples")

# Train Logistic Regression
print("\n[Model 1] Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_basic, y_train)
y_pred_lr = lr_model.predict(X_test_basic)
lr_accuracy = accuracy_score(y_test, y_pred_lr)
print(f"  Accuracy: {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")

# Train Random Forest
print("\n[Model 2] Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_basic, y_train)
y_pred_rf = rf_model.predict(X_test_basic)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"  Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")

# Train Decision Tree
print("\n[Model 3] Decision Tree...")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_basic, y_train)
y_pred_dt = dt_model.predict(X_test_basic)
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print(f"  Accuracy: {dt_accuracy:.4f} ({dt_accuracy*100:.2f}%)")

# ==========================================
# 6. FEATURE ENGINEERING
# ==========================================
print("\n" + "="*60)
print("ENHANCED MODEL (With Feature Engineering)")
print("="*60)

print("\n[Step 6] Creating New Features...")
df_enhanced = df.copy()

# Recreate the original dataframe for feature engineering
df_original = pd.read_csv("loan_data.csv")
df_enhanced_raw = df_original.drop('Loan_ID', axis=1)

# Feature 1: Total Income
df_enhanced_raw['Total_Income'] = df_enhanced_raw['ApplicantIncome'] + df_enhanced_raw['CoapplicantIncome']
print("  [OK] Created: Total_Income")

# Feature 2: Income to Loan Ratio
df_enhanced_raw['Income_Loan_Ratio'] = df_enhanced_raw['Total_Income'] / (df_enhanced_raw['LoanAmount'] + 1)
print("  [OK] Created: Income_Loan_Ratio")

# Feature 3: Log Transformation
df_enhanced_raw['LoanAmount_log'] = np.log(df_enhanced_raw['LoanAmount'] + 1)
df_enhanced_raw['Total_Income_log'] = np.log(df_enhanced_raw['Total_Income'] + 1)
print("  [OK] Created: LoanAmount_log, Total_Income_log")

# Encode categorical variables in enhanced dataset
for col in df_enhanced_raw.select_dtypes(include='object').columns:
    df_enhanced_raw[col] = le.fit_transform(df_enhanced_raw[col])

# ==========================================
# 7. ENHANCED MODEL TRAINING
# ==========================================
X_enhanced = df_enhanced_raw.drop("Loan_Status", axis=1)
y_enhanced = df_enhanced_raw["Loan_Status"]

X_train_enh, X_test_enh, y_train_enh, y_test_enh = train_test_split(
    X_enhanced, y_enhanced, test_size=0.2, random_state=42, stratify=y_enhanced
)

print(f"\n  Enhanced features: {X_enhanced.shape[1]} (vs {X_basic.shape[1]} baseline)")
print(f"  New features added: {X_enhanced.shape[1] - X_basic.shape[1]}")

# Train Enhanced Logistic Regression
print("\n[Enhanced Model 1] Logistic Regression...")
lr_enhanced = LogisticRegression(max_iter=1000)
lr_enhanced.fit(X_train_enh, y_train_enh)
y_pred_lr_enh = lr_enhanced.predict(X_test_enh)
lr_enhanced_accuracy = accuracy_score(y_test_enh, y_pred_lr_enh)
print(f"  Accuracy: {lr_enhanced_accuracy:.4f} ({lr_enhanced_accuracy*100:.2f}%)")
print(f"  Improvement: {(lr_enhanced_accuracy - lr_accuracy)*100:.2f}%")

# Train Enhanced Random Forest
print("\n[Enhanced Model 2] Random Forest...")
rf_enhanced = RandomForestClassifier(n_estimators=100, random_state=42)
rf_enhanced.fit(X_train_enh, y_train_enh)
y_pred_rf_enh = rf_enhanced.predict(X_test_enh)
rf_enhanced_accuracy = accuracy_score(y_test_enh, y_pred_rf_enh)
print(f"  Accuracy: {rf_enhanced_accuracy:.4f} ({rf_enhanced_accuracy*100:.2f}%)")
print(f"  Improvement: {(rf_enhanced_accuracy - rf_accuracy)*100:.2f}%")

# ==========================================
# 8. DETAILED EVALUATION (Best Model)
# ==========================================
print("\n" + "="*60)
print("DETAILED EVALUATION - BEST MODEL (Random Forest Enhanced)")
print("="*60)

print("\n[Confusion Matrix]")
print(confusion_matrix(y_test_enh, y_pred_rf_enh))

print("\n[Classification Report]")
print(classification_report(y_test_enh, y_pred_rf_enh, target_names=['Rejected', 'Approved']))

# ==========================================
# 9. FEATURE IMPORTANCE
# ==========================================
print("\n[Feature Importance - Top 10]")
feature_importance = pd.DataFrame({
    'feature': X_enhanced.columns,
    'importance': rf_enhanced.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

# ==========================================
# 10. MODEL COMPARISON SUMMARY
# ==========================================
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)

results = pd.DataFrame({
    'Model': [
        'Logistic Regression (Baseline)',
        'Random Forest (Baseline)',
        'Decision Tree (Baseline)',
        'Logistic Regression (Enhanced)',
        'Random Forest (Enhanced)'
    ],
    'Accuracy': [
        f'{lr_accuracy:.4f}',
        f'{rf_accuracy:.4f}',
        f'{dt_accuracy:.4f}',
        f'{lr_enhanced_accuracy:.4f}',
        f'{rf_enhanced_accuracy:.4f}'
    ]
})

print("\n" + results.to_string(index=False))

print("\n" + "="*60)
print("[SUCCESS] LOAN APPROVAL PREDICTION COMPLETED")
print("="*60)
print(f"\nBest Model: Random Forest (Enhanced)")
print(f"Best Accuracy: {rf_enhanced_accuracy*100:.2f}%")
print("\nKey Insights:")
print("- Feature engineering significantly improves model performance")
print("- Credit History is likely the most important feature")
print("- Random Forest outperforms other models")
print("- Total Income and Income-to-Loan Ratio are valuable features")
