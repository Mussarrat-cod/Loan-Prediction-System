"""
Improved Loan Prediction Model Training
========================================
This script uses advanced techniques to improve model accuracy:
- Hyperparameter tuning
- Cross-validation
- Feature engineering
- Multiple algorithms
- Ensemble methods
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("IMPROVED LOAN PREDICTION MODEL TRAINING")
print("="*70)

# Load dataset
print("\n[1/8] Loading and analyzing dataset...")
df = pd.read_csv("loan_data.csv")
print(f"✓ Loaded {len(df)} records")
print(f"✓ Features: {len(df.columns)}")

# Prepare data
df_original = df.copy()
df = df.drop('Loan_ID', axis=1)

# Handle missing values and clean data
print("\n[2/8] Handling missing values...")
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
print("\n[3/8] Advanced feature engineering...")
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

print(f"✓ Created {len(df_features.columns) - len(df.columns)} new features")

# Create label encoders
print("\n[4/8] Encoding categorical variables...")
encoders = {}
categorical_cols = df_features.select_dtypes(include='object').columns

for col in categorical_cols:
    encoders[col] = LabelEncoder()
    df_features[col] = encoders[col].fit_transform(df_features[col])

print(f"✓ Encoded {len(categorical_cols)} categorical columns")

# Feature selection
print("\n[5/8] Feature selection...")
X = df_features.drop("Loan_Status", axis=1)
y = df_features["Loan_Status"]

# Select best features
selector = SelectKBest(score_func=f_classif, k='all')
X_selected = selector.fit_transform(X, y)
feature_scores = pd.DataFrame({
    'feature': X.columns,
    'score': selector.scores_
}).sort_values('score', ascending=False)

print("Top 10 features by importance:")
for idx, row in feature_scores.head(10).iterrows():
    print(f"  {row['feature']}: {row['score']:.2f}")

# Keep only features with good scores
good_features = feature_scores[feature_scores['score'] > 1]['feature'].tolist()
X = X[good_features]
print(f"✓ Selected {len(good_features)} features")

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features for algorithms that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Split data: {len(X_train)} train, {len(X_test)} test samples")

# Hyperparameter tuning and model comparison
print("\n[6/8] Training multiple models with hyperparameter tuning...")

# Random Forest with GridSearch
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, 
                       cv=5, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
rf_score = rf_grid.best_score_

# Gradient Boosting
gb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}

gb_grid = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_params,
                       cv=5, scoring='accuracy', n_jobs=-1)
gb_grid.fit(X_train, y_train)
best_gb = gb_grid.best_estimator_
gb_score = gb_grid.best_score_

# Logistic Regression
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_cv_score = cross_val_score(lr_model, X_train_scaled, y_train, cv=5).mean()

print(f"✓ Random Forest CV Score: {rf_score:.4f}")
print(f"✓ Gradient Boosting CV Score: {gb_score:.4f}")
print(f"✓ Logistic Regression CV Score: {lr_cv_score:.4f}")

# Create ensemble model
print("\n[7/8] Creating ensemble model...")
ensemble = VotingClassifier(
    estimators=[
        ('rf', best_rf),
        ('gb', best_gb),
        ('lr', LogisticRegression(random_state=42, max_iter=1000))
    ],
    voting='soft'
)

# Train ensemble
ensemble.fit(X_train, y_train)
ensemble_score = cross_val_score(ensemble, X_train, y_train, cv=5).mean()
print(f"✓ Ensemble CV Score: {ensemble_score:.4f}")

# Final evaluation
print("\n[8/8] Final model evaluation...")
models = {
    'Random Forest': best_rf,
    'Gradient Boosting': best_gb,
    'Ensemble': ensemble
}

best_model = None
best_accuracy = 0
best_name = ""

for name, model in models.items():
    if name == 'Logistic Regression':
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{name}:")
    print(f"  Test Accuracy: {accuracy*100:.2f}%")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_name = name

print(f"\n🏆 Best model: {best_name} with {best_accuracy*100:.2f}% accuracy")

# Save the best model and components
print("\n" + "="*70)
print("SAVING IMPROVED MODEL")
print("="*70)

joblib.dump(best_model, 'model.pkl')
joblib.dump(encoders, 'encoders.pkl')
joblib.dump(list(X.columns), 'feature_names.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Save comprehensive model info
model_info = {
    'best_model': best_name,
    'accuracy': float(best_accuracy),
    'n_samples_train': len(X_train),
    'n_samples_test': len(X_test),
    'n_features': len(X.columns),
    'feature_scores': feature_scores.head(15).to_dict('records'),
    'model_type': type(best_model).__name__,
    'cross_val_scores': {
        'random_forest': float(rf_score),
        'gradient_boosting': float(gb_score),
        'ensemble': float(ensemble_score)
    },
    'hyperparameters': best_model.get_params() if hasattr(best_model, 'get_params') else {},
    'feature_importance': []
}

# Add feature importance if available
if hasattr(best_model, 'feature_importances_'):
    model_info['feature_importance'] = [
        {'feature': feat, 'importance': float(imp)}
        for feat, imp in zip(X.columns, best_model.feature_importances_)
    ]

with open('model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("✓ Saved improved model.pkl")
print("✓ Saved encoders.pkl")
print("✓ Saved feature_names.pkl")
print("✓ Saved scaler.pkl")
print("✓ Saved model_info.json")

print("\n" + "="*70)
print("IMPROVED MODEL TRAINING COMPLETE!")
print("="*70)
print(f"\n🎯 Final Accuracy: {best_accuracy*100:.2f}%")
print(f"🏆 Best Model: {best_name}")
print(f"📊 Features Used: {len(X.columns)}")
print("\nReady to use in Flask API!")
