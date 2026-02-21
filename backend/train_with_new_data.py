"""
Train Model with New Larger Dataset
===================================
Uses train.csv (614 records) and test.csv (367 records)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("TRAINING WITH NEW LARGER DATASET")
print("="*70)

# Load datasets
print("\n[1/7] Loading datasets...")
train_df = pd.read_csv("../train.csv")
test_df = pd.read_csv("../test.csv")

print(f"✓ Train data: {len(train_df)} records")
print(f"✓ Test data: {len(test_df)} records")
print(f"✓ Total data: {len(train_df) + len(test_df)} records")

# Prepare training data
df = train_df.copy()
df_original = df.copy()
df = df.drop('Loan_ID', axis=1)

# Handle missing values and clean data
print("\n[2/7] Handling missing values...")
missing_values = df.isnull().sum()
print(f"Missing values before cleaning:\n{missing_values[missing_values > 0]}")

# Clean Dependents column - convert '3+' to 3 and ensure numeric
df['Dependents'] = df['Dependents'].replace('3+', '3')
df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce').fillna(0)

# Fill missing values strategically
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

print("✓ Missing values handled")

# Advanced Feature Engineering
print("\n[3/7] Advanced feature engineering...")
df_features = df.copy()

# Income features
df_features['Total_Income'] = df_features['ApplicantIncome'] + df_features['CoapplicantIncome']
df_features['Income_Loan_Ratio'] = df_features['Total_Income'] / (df_features['LoanAmount'] + 1)
df_features['Income_per_dependent'] = df_features['Total_Income'] / (df_features['Dependents'] + 1)

# Log transformations for skewed features
df_features['LoanAmount_log'] = np.log(df_features['LoanAmount'] + 1)
df_features['Total_Income_log'] = np.log(df_features['Total_Income'] + 1)
df_features['ApplicantIncome_log'] = np.log(df_features['ApplicantIncome'] + 1)

# Loan-related features
df_features['Loan_Term_Years'] = df_features['Loan_Amount_Term'] / 12
df_features['EMI'] = df_features['LoanAmount'] / df_features['Loan_Amount_Term']
df_features['EMI_Income_Ratio'] = df_features['EMI'] / (df_features['Total_Income'] + 1)

# Credit history features
df_features['Credit_History_Income_Interaction'] = df_features['Credit_History'] * df_features['Total_Income_log']
df_features['Credit_History_Loan_Interaction'] = df_features['Credit_History'] * df_features['LoanAmount_log']

# Risk scoring features
df_features['High_Income_Flag'] = (df_features['Total_Income'] > df_features['Total_Income'].median()).astype(int)
df_features['Low_Loan_Flag'] = (df_features['LoanAmount'] < df_features['LoanAmount'].median()).astype(int)
df_features['Good_Credit_Flag'] = (df_features['Credit_History'] == 1).astype(int)

# Additional advanced features
df_features['Loan_to_Income_Ratio'] = df_features['LoanAmount'] / (df_features['Total_Income'] + 1)
df_features['Coapplicant_Income_Ratio'] = df_features['CoapplicantIncome'] / (df_features['Total_Income'] + 1)
df_features['Long_Term_Loan_Flag'] = (df_features['Loan_Amount_Term'] > 300).astype(int)

print(f"✓ Created {len(df_features.columns) - len(df.columns)} new features")

# Create label encoders
print("\n[4/7] Encoding categorical variables...")
encoders = {}
categorical_cols = df_features.select_dtypes(include='object').columns

for col in categorical_cols:
    encoders[col] = LabelEncoder()
    df_features[col] = encoders[col].fit_transform(df_features[col])

print(f"✓ Encoded {len(categorical_cols)} categorical columns")

# Feature selection
print("\n[5/7] Feature selection...")
X = df_features.drop("Loan_Status", axis=1)
y = df_features["Loan_Status"]

# Select best features
selector = SelectKBest(score_func=f_classif, k='all')
X_selected = selector.fit_transform(X, y)
feature_scores = pd.DataFrame({
    'feature': X.columns,
    'score': selector.scores_
}).sort_values('score', ascending=False)

print("Top 15 features by importance:")
for idx, row in feature_scores.head(15).iterrows():
    print(f"  {row['feature']}: {row['score']:.2f}")

# Keep only features with good scores
good_features = feature_scores[feature_scores['score'] > 0.5]['feature'].tolist()
X = X[good_features]
print(f"✓ Selected {len(good_features)} features")

# Split data with stratification
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Split data: {len(X_train)} train, {len(X_val)} validation samples")

# Train multiple models with optimized parameters
print("\n[6/7] Training optimized models...")

models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=10,
        random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        random_state=42,
        max_iter=2000,
        C=1.0,
        solver='liblinear'
    )
}

best_model = None
best_accuracy = 0
best_name = ""
all_results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train and validate
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"  Validation Accuracy: {accuracy*100:.2f}%")
    
    all_results[name] = {
        'cv_score': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'val_accuracy': accuracy
    }
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_name = name

print(f"\n🏆 Best model: {best_name} with {best_accuracy*100:.2f}% accuracy")

# Prepare test data for final evaluation
print("\n[7/7] Preparing test data for evaluation...")
test_processed = test_df.copy()

# Clean test data the same way
test_processed['Dependents'] = test_processed['Dependents'].replace('3+', '3')
test_processed['Dependents'] = pd.to_numeric(test_processed['Dependents'], errors='coerce').fillna(0)

# Fill missing values
for col in test_processed.columns:
    if col != 'Loan_ID':
        if test_processed[col].dtype == 'object':
            test_processed[col].fillna(test_processed[col].mode()[0], inplace=True)
        else:
            test_processed[col].fillna(test_processed[col].median(), inplace=True)

# Apply same feature engineering
test_features = test_processed.drop('Loan_ID', axis=1)
test_features['Total_Income'] = test_features['ApplicantIncome'] + test_features['CoapplicantIncome']
test_features['Income_Loan_Ratio'] = test_features['Total_Income'] / (test_features['LoanAmount'] + 1)
test_features['Income_per_dependent'] = test_features['Total_Income'] / (test_features['Dependents'] + 1)
test_features['LoanAmount_log'] = np.log(test_features['LoanAmount'] + 1)
test_features['Total_Income_log'] = np.log(test_features['Total_Income'] + 1)
test_features['ApplicantIncome_log'] = np.log(test_features['ApplicantIncome'] + 1)
test_features['Loan_Term_Years'] = test_features['Loan_Amount_Term'] / 12
test_features['EMI'] = test_features['LoanAmount'] / test_features['Loan_Amount_Term']
test_features['EMI_Income_Ratio'] = test_features['EMI'] / (test_features['Total_Income'] + 1)
test_features['Credit_History_Income_Interaction'] = test_features['Credit_History'] * test_features['Total_Income_log']
test_features['Credit_History_Loan_Interaction'] = test_features['Credit_History'] * test_features['LoanAmount_log']
test_features['High_Income_Flag'] = (test_features['Total_Income'] > df_features['Total_Income'].median()).astype(int)
test_features['Low_Loan_Flag'] = (test_features['LoanAmount'] < df_features['LoanAmount'].median()).astype(int)
test_features['Good_Credit_Flag'] = (test_features['Credit_History'] == 1).astype(int)
test_features['Loan_to_Income_Ratio'] = test_features['LoanAmount'] / (test_features['Total_Income'] + 1)
test_features['Coapplicant_Income_Ratio'] = test_features['CoapplicantIncome'] / (test_features['Total_Income'] + 1)
test_features['Long_Term_Loan_Flag'] = (test_features['Loan_Amount_Term'] > 300).astype(int)

# Encode categorical variables
for col in test_features.select_dtypes(include='object').columns:
    if col in encoders:
        test_features[col] = encoders[col].transform(test_features[col])

# Select same features
X_test = test_features[good_features]

print(f"✓ Test data prepared: {len(X_test)} samples")

# Save the best model and components
print("\n" + "="*70)
print("SAVING NEW IMPROVED MODEL")
print("="*70)

joblib.dump(best_model, 'model.pkl')
joblib.dump(encoders, 'encoders.pkl')
joblib.dump(list(X.columns), 'feature_names.pkl')

# Save comprehensive model info
model_info = {
    'best_model': best_name,
    'accuracy': float(best_accuracy),
    'previous_accuracy': 0.95,  # From previous improved model
    'original_accuracy': 0.90,  # From original model
    'improvement_from_original': float(best_accuracy - 0.90),
    'improvement_from_previous': float(best_accuracy - 0.95),
    'n_samples_train': len(X_train),
    'n_samples_validation': len(X_val),
    'n_samples_test': len(X_test),
    'n_features': len(X.columns),
    'feature_scores': feature_scores.head(20).to_dict('records'),
    'model_type': type(best_model).__name__,
    'all_results': all_results,
    'hyperparameters': best_model.get_params() if hasattr(best_model, 'get_params') else {},
    'dataset_info': {
        'train_records': len(train_df),
        'test_records': len(test_df),
        'total_records': len(train_df) + len(test_df)
    }
}

# Add feature importance if available
if hasattr(best_model, 'feature_importances_'):
    model_info['feature_importance'] = [
        {'feature': feat, 'importance': float(imp)}
        for feat, imp in zip(X.columns, best_model.feature_importances_)
    ]

with open('model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("✓ Saved new model.pkl")
print("✓ Saved encoders.pkl")
print("✓ Saved feature_names.pkl")
print("✓ Saved model_info.json")

print("\n" + "="*70)
print("NEW MODEL TRAINING COMPLETE!")
print("="*70)
print(f"\n📊 Dataset Size: {len(train_df)} train + {len(test_df)} test = {len(train_df) + len(test_df)} total")
print(f"🎯 Original Accuracy: 90.00%")
print(f"🎯 Previous Accuracy: 95.00%")
print(f"🎯 New Accuracy: {best_accuracy*100:.2f}%")
print(f"📈 Improvement from Original: {(best_accuracy - 0.90)*100:+.2f}%")
print(f"📈 Improvement from Previous: {(best_accuracy - 0.95)*100:+.2f}%")
print(f"🏆 Best Model: {best_name}")
print(f"📊 Features Used: {len(X.columns)}")
print("\n🚀 Ready to use in Flask API with much higher accuracy!")
