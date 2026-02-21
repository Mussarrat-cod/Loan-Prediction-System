"""
Ultra-Optimized Loan Prediction Model
=====================================
Maximum accuracy optimization with advanced techniques
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.neural_network import MLPClassifier
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ULTRA-OPTIMIZED LOAN PREDICTION MODEL")
print("="*70)

# Load datasets
print("\n[1/8] Loading and analyzing datasets...")
train_df = pd.read_csv("../train.csv")
test_df = pd.read_csv("../test.csv")

print(f"✓ Train data: {len(train_df)} records")
print(f"✓ Test data: {len(test_df)} records")

# Prepare training data
df = train_df.copy()

# Advanced data cleaning and preprocessing
print("\n[2/8] Advanced data preprocessing...")

# Clean Dependents column
df['Dependents'] = df['Dependents'].replace('3+', '3')
df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce').fillna(0)

# Handle missing values with advanced strategies
for col in df.columns:
    if col == 'Loan_Status':
        continue
    if df[col].dtype == 'object':
        # For categorical, use mode
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        # For numerical, use median with some intelligence
        if col in ['LoanAmount', 'ApplicantIncome', 'CoapplicantIncome']:
            # Use median by education level for income-related columns
            for edu_level in df['Education'].unique():
                if edu_level in df['Education'].values:
                    median_val = df[df['Education'] == edu_level][col].median()
                    df.loc[(df[col].isnull()) & (df['Education'] == edu_level), col] = median_val
        df[col].fillna(df[col].median(), inplace=True)

print("✓ Advanced missing value handling completed")

# Ultra-advanced feature engineering
print("\n[3/8] Ultra-advanced feature engineering...")
df_features = df.copy()

# Income-based features
df_features['Total_Income'] = df_features['ApplicantIncome'] + df_features['CoapplicantIncome']
df_features['Income_per_Dependent'] = df_features['Total_Income'] / (df_features['Dependents'] + 1)
df_features['Coapplicant_Ratio'] = df_features['CoapplicantIncome'] / (df_features['Total_Income'] + 1)

# Loan-related features
df_features['Loan_to_Income'] = df_features['LoanAmount'] / (df_features['Total_Income'] + 1)
df_features['EMI'] = df_features['LoanAmount'] / df_features['Loan_Amount_Term']
df_features['EMI_to_Income'] = df_features['EMI'] / (df_features['Total_Income'] + 1)
df_features['Loan_Term_Years'] = df_features['Loan_Amount_Term'] / 12

# Log transformations for skewed data
df_features['LoanAmount_log'] = np.log1p(df_features['LoanAmount'])
df_features['Total_Income_log'] = np.log1p(df_features['Total_Income'])
df_features['ApplicantIncome_log'] = np.log1p(df_features['ApplicantIncome'])
df_features['CoapplicantIncome_log'] = np.log1p(df_features['CoapplicantIncome'])

# Credit history interactions
df_features['Credit_x_Income'] = df_features['Credit_History'] * df_features['Total_Income_log']
df_features['Credit_x_Loan'] = df_features['Credit_History'] * df_features['LoanAmount_log']
df_features['Credit_x_EMI'] = df_features['Credit_History'] * df_features['EMI_to_Income']

# Risk assessment features
income_median = df_features['Total_Income'].median()
loan_median = df_features['LoanAmount'].median()
df_features['High_Income'] = (df_features['Total_Income'] > income_median).astype(int)
df_features['Low_Loan'] = (df_features['LoanAmount'] < loan_median).astype(int)
df_features['Good_Credit'] = (df_features['Credit_History'] == 1).astype(int)

# Combined risk score
df_features['Risk_Score'] = (
    df_features['Good_Credit'] * 3 +
    df_features['High_Income'] * 2 +
    df_features['Low_Loan'] * 1 +
    (df_features['Education'] == 0) * 1  # Graduate = 0, Not Graduate = 1
)

# Property area encoding (Urban=2, Semiurban=1, Rural=0)
property_mapping = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}
df_features['Property_Area_Score'] = df_features['Property_Area'].map(property_mapping)

# Interaction features
df_features['Income_x_Property'] = df_features['Total_Income_log'] * df_features['Property_Area_Score']
df_features['Loan_x_Property'] = df_features['LoanAmount_log'] * df_features['Property_Area_Score']

print(f"✓ Created {len(df_features.columns) - len(df.columns)} advanced features")

# Encode categorical variables
print("\n[4/8] Smart categorical encoding...")
encoders = {}
categorical_cols = df_features.select_dtypes(include='object').columns

for col in categorical_cols:
    if col != 'Loan_Status':
        encoders[col] = LabelEncoder()
        df_features[col] = encoders[col].fit_transform(df_features[col])

# Encode target variable
df_features['Loan_Status'] = df_features['Loan_Status'].map({'Y': 1, 'N': 0})

print(f"✓ Encoded {len(categorical_cols)} categorical columns")

# Feature selection with multiple methods
print("\n[5/8] Multi-method feature selection...")
X = df_features.drop("Loan_Status", axis=1)
y = df_features["Loan_Status"]

# Method 1: Statistical tests
selector_f = SelectKBest(score_func=f_classif, k='all')
X_f = selector_f.fit_transform(X, y)
f_scores = pd.DataFrame({
    'feature': X.columns,
    'f_score': selector_f.scores_
})

# Method 2: Feature importance with tree
rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
rf_temp.fit(X, y)
importance_scores = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_temp.feature_importances_
})

# Combine scores
feature_scores = f_scores.merge(importance_scores, on='feature')
feature_scores['combined_score'] = feature_scores['f_score'] * feature_scores['importance']
feature_scores = feature_scores.sort_values('combined_score', ascending=False)

print("Top 20 features by combined importance:")
for idx, row in feature_scores.head(20).iterrows():
    print(f"  {row['feature']}: {row['combined_score']:.2f}")

# Select top features
top_features = feature_scores.head(20)['feature'].tolist()
X = X[top_features]
print(f"✓ Selected {len(top_features)} best features")

# Advanced cross-validation and model training
print("\n[6/8] Advanced model training with cross-validation...")

# Define models with optimized parameters
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=500,
        max_depth=25,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    ),
    'Extra Trees': ExtraTreesClassifier(
        n_estimators=400,
        max_depth=20,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        C=2.0,
        penalty='l2',
        random_state=42,
        max_iter=2000,
        solver='liblinear'
    ),
    'Neural Network': MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=1000,
        random_state=42
    )
}

# Stratified cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_model = None
best_accuracy = 0
best_name = ""
all_results = {}

for name, model in models.items():
    print(f"\n🔧 Training {name}...")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train on full dataset
    model.fit(X, y)
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        top_importance = sorted(zip(X.columns, model.feature_importances_), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top features: {[f'{feat}({imp:.3f})' for feat, imp in top_importance]}")
    
    all_results[name] = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'cv_min': cv_scores.min(),
        'cv_max': cv_scores.max()
    }
    
    if cv_scores.mean() > best_accuracy:
        best_accuracy = cv_scores.mean()
        best_model = model
        best_name = name

print(f"\n🏆 Best model: {best_name} with {best_accuracy*100:.2f}% CV accuracy")

# Final validation with train-test split
print("\n[7/8] Final validation...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_val)
final_accuracy = accuracy_score(y_val, y_pred)
print(f"Final validation accuracy: {final_accuracy*100:.2f}%")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# Save the ultra-optimized model
print("\n[8/8] Saving ultra-optimized model...")
joblib.dump(best_model, 'model.pkl')
joblib.dump(encoders, 'encoders.pkl')
joblib.dump(list(X.columns), 'feature_names.pkl')

# Save comprehensive model info
model_info = {
    'best_model': best_name,
    'cv_accuracy': float(best_accuracy),
    'validation_accuracy': float(final_accuracy),
    'previous_accuracy': 0.8862,
    'improvement': float(final_accuracy - 0.8862),
    'n_samples': len(X),
    'n_features': len(X.columns),
    'feature_scores': feature_scores.head(25).to_dict('records'),
    'model_type': type(best_model).__name__,
    'all_results': all_results,
    'hyperparameters': best_model.get_params() if hasattr(best_model, 'get_params') else {},
    'dataset_info': {
        'train_records': len(train_df),
        'test_records': len(test_df),
        'total_records': len(train_df) + len(test_df)
    }
}

if hasattr(best_model, 'feature_importances_'):
    model_info['feature_importance'] = [
        {'feature': feat, 'importance': float(imp)}
        for feat, imp in zip(X.columns, best_model.feature_importances_)
    ]

with open('model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("✓ Saved ultra-optimized model.pkl")
print("✓ Saved encoders.pkl") 
print("✓ Saved feature_names.pkl")
print("✓ Saved model_info.json")

print("\n" + "="*70)
print("ULTRA-OPTIMIZED MODEL TRAINING COMPLETE!")
print("="*70)
print(f"\n🚀 Cross-Validation Accuracy: {best_accuracy*100:.2f}%")
print(f"🎯 Validation Accuracy: {final_accuracy*100:.2f}%")
print(f"📈 Improvement: {(final_accuracy - 0.8862)*100:+.2f}%")
print(f"🏆 Best Model: {best_name}")
print(f"🧠 Features Used: {len(X.columns)}")
print(f"📊 Training Records: {len(X)}")
print("\n🔥 ULTRA-OPTIMIZED MODEL READY FOR PRODUCTION!")
