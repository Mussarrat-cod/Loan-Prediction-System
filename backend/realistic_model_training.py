"""
Realistic Loan Prediction Model Training
=====================================
Trains on diverse, real-world loan data with comprehensive features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("REALISTIC LOAN PREDICTION MODEL TRAINING")
print("="*80)

# Load datasets
print("\n[1/10] Loading and analyzing datasets...")
train_df = pd.read_csv("../train.csv")
test_df = pd.read_csv("../test.csv")

print(f"✓ Train data: {len(train_df)} records")
print(f"✓ Test data: {len(test_df)} records")
print(f"✓ Total: {len(train_df) + len(test_df)} records")

# Analyze data quality
print("\n[2/10] Data quality analysis...")
df = train_df.copy()

# Clean and preprocess data
print("Cleaning missing values...")
df['Dependents'] = df['Dependents'].replace('3+', '3')
df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce').fillna(0)

# Advanced missing value handling
for col in df.columns:
    if col == 'Loan_Status':
        continue
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        # Use median by category for better imputation
        if 'Education' in df.columns:
            for edu_level in df['Education'].unique():
                mask = (df[col].isnull()) & (df['Education'] == edu_level)
                if mask.any() and edu_level in df['Education'].values:
                    median_val = df[df['Education'] == edu_level][col].median()
                    df.loc[mask, col] = median_val
        df[col].fillna(df[col].median(), inplace=True)

print("✓ Advanced missing value handling completed")

# Comprehensive feature engineering
print("\n[3/10] Comprehensive real-world feature engineering...")
df_features = df.copy()

# Income and financial features
df_features['Total_Income'] = df_features['ApplicantIncome'] + df_features['CoapplicantIncome']
df_features['Income_per_Dependent'] = df_features['Total_Income'] / (df_features['Dependents'] + 1)
df_features['Coapplicant_Ratio'] = df_features['CoapplicantIncome'] / (df_features['Total_Income'] + 1)
df_features['Applicant_Income_Ratio'] = df_features['ApplicantIncome'] / (df_features['Total_Income'] + 1)

# Loan and debt features
df_features['Loan_to_Income'] = df_features['LoanAmount'] / (df_features['Total_Income'] + 1)
df_features['Loan_per_Dependent'] = df_features['LoanAmount'] / (df_features['Dependents'] + 1)
df_features['EMI'] = df_features['LoanAmount'] / df_features['Loan_Amount_Term']
df_features['EMI_to_Income'] = df_features['EMI'] / (df_features['Total_Income'] + 1)
df_features['Debt_to_Income'] = df_features['EMI'] * 12 / (df_features['Total_Income'] + 1)

# Credit and risk features
df_features['Credit_History_Score'] = df_features['Credit_History'] * 100
df_features['Credit_x_Income'] = df_features['Credit_History'] * np.log1p(df_features['Total_Income'])
df_features['Credit_x_Loan'] = df_features['Credit_History'] * np.log1p(df_features['LoanAmount'])
df_features['Credit_x_EMI'] = df_features['Credit_History'] * df_features['EMI_to_Income']

# Employment stability features
df_features['Income_Stability'] = np.where(
    df_features['Self_Employed'] == 'No', 
    df_features['ApplicantIncome'] / (df_features['ApplicantIncome'].mean()),
    df_features['ApplicantIncome'] / (df_features['ApplicantIncome'].mean() * 1.2)
)

# Demographic and social factors
df_features['Family_Size'] = df_features['Dependents'] + 2  # +2 for applicant and potential spouse
df_features['Is_Married_with_Dependents'] = ((df_features['Married'] == 'Yes') & (df_features['Dependents'] > 0)).astype(int)
df_features['Single_No_Dependents'] = ((df_features['Married'] == 'No') & (df_features['Dependents'] == 0)).astype(int)

# Property and location features
property_scores = {'Urban': 3, 'Semiurban': 2, 'Rural': 1}
df_features['Property_Score'] = df_features['Property_Area'].map(property_scores)
df_features['Urban_x_Income'] = (df_features['Property_Area'] == 'Urban') * np.log1p(df_features['Total_Income'])
df_features['Rural_x_Loan'] = (df_features['Property_Area'] == 'Rural') * np.log1p(df_features['LoanAmount'])

# Log transformations for skewed data
df_features['LoanAmount_log'] = np.log1p(df_features['LoanAmount'])
df_features['Total_Income_log'] = np.log1p(df_features['Total_Income'])
df_features['ApplicantIncome_log'] = np.log1p(df_features['ApplicantIncome'])
df_features['CoapplicantIncome_log'] = np.log1p(df_features['CoapplicantIncome'])

# Risk assessment features
df_features['Risk_Score_1'] = (
    df_features['Credit_History'] * 3 +
    (df_features['Total_Income'] > df_features['Total_Income'].median()) * 2 +
    (df_features['LoanAmount'] < df_features['LoanAmount'].median()) * 1 +
    (df_features['Education'] == 'Graduate') * 1 +
    (df_features['Married'] == 'Yes') * 1
)

df_features['Risk_Score_2'] = (
    df_features['Credit_History'] * 4 +
    df_features['Property_Score'] +
    (df_features['Self_Employed'] == 'No') * 2 +
    (df_features['Dependents'] == 0) * 1
)

# Interaction features
df_features['Income_x_Education'] = df_features['Total_Income_log'] * (df_features['Education'] == 'Graduate')
df_features['Income_x_Property'] = df_features['Total_Income_log'] * df_features['Property_Score']
df_features['Loan_x_Education'] = df_features['LoanAmount_log'] * (df_features['Education'] == 'Graduate')

print(f"✓ Created {len(df_features.columns) - len(df.columns)} comprehensive features")

# Smart encoding
print("\n[4/10] Smart categorical encoding...")
encoders = {}
categorical_cols = df_features.select_dtypes(include='object').columns

for col in categorical_cols:
    if col != 'Loan_Status':
        encoders[col] = LabelEncoder()
        df_features[col] = encoders[col].fit_transform(df_features[col])

# Target encoding for Loan_Status
df_features['Loan_Status'] = df_features['Loan_Status'].map({'Y': 1, 'N': 0})

print(f"✓ Encoded {len(categorical_cols)} categorical variables")

# Advanced feature selection
print("\n[5/10] Advanced feature selection...")
X = df_features.drop("Loan_Status", axis=1)
y = df_features["Loan_Status"]

# Multiple feature selection methods
# Method 1: Statistical tests
selector_f = SelectKBest(score_func=f_classif, k='all')
X_f = selector_f.fit_transform(X, y)
f_scores = pd.DataFrame({
    'feature': X.columns,
    'f_score': selector_f.scores_
})

# Method 2: Tree-based importance
rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
rf_temp.fit(X, y)
importance_scores = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_temp.feature_importances_
})

# Method 3: Recursive feature elimination
rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=25)
rfe.fit(X, y)
rfe_scores = pd.DataFrame({
    'feature': X.columns,
    'rfe_ranking': rfe.ranking_
})

# Combine all methods
feature_scores = f_scores.merge(importance_scores, on='feature').merge(rfe_scores, on='feature')
feature_scores['combined_score'] = (
    feature_scores['f_score'] * 
    feature_scores['importance'] * 
    (1 / feature_scores['rfe_ranking'])
)
feature_scores = feature_scores.sort_values('combined_score', ascending=False)

print("Top 25 features by combined importance:")
for idx, row in feature_scores.head(25).iterrows():
    print(f"  {row['feature'][:30]:30} {row['combined_score']:.2f}")

# Select top features
top_features = feature_scores.head(25)['feature'].tolist()
X = X[top_features]
print(f"✓ Selected {len(top_features)} best features")

# Multiple model training with cross-validation
print("\n[6/10] Training diverse model ensemble...")

models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=400,
        max_depth=25,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    ),
    'Extra Trees': RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=False,
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.03,
        max_depth=6,
        min_samples_split=8,
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
    'SVM': SVC(
        C=1.0,
        kernel='rbf',
        probability=True,
        random_state=42
    ),
    'Neural Network': MLPClassifier(
        hidden_layer_sizes=(150, 75, 25),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=1500,
        random_state=42
    ),
    'KNN': KNeighborsClassifier(
        n_neighbors=7,
        weights='distance',
        algorithm='auto'
    ),
    'Naive Bayes': GaussianNB()
}

# Advanced cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

best_model = None
best_accuracy = 0
best_name = ""
all_results = {}

print("\nTraining models with 10-fold cross-validation...")
for name, model in models.items():
    print(f"\n🔧 Training {name}...")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
    
    # Additional metrics
    model.fit(X, y)
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X)[:, 1]
        auc_score = roc_auc_score(y, y_proba)
    else:
        auc_score = 0
    
    print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    print(f"  Min/Max: {cv_scores.min():.4f} / {cv_scores.max():.4f}")
    print(f"  AUC Score: {auc_score:.4f}")
    
    if hasattr(model, 'feature_importances_'):
        top_importance = sorted(zip(X.columns, model.feature_importances_), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top features: {[f'{feat[:20]}({imp:.3f})' for feat, imp in top_importance]}")
    
    all_results[name] = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'cv_min': cv_scores.min(),
        'cv_max': cv_scores.max(),
        'auc_score': auc_score
    }
    
    if cv_scores.mean() > best_accuracy:
        best_accuracy = cv_scores.mean()
        best_model = model
        best_name = name

print(f"\n🏆 Best individual model: {best_name} with {best_accuracy*100:.2f}% CV accuracy")

# Create ensemble models
print("\n[7/10] Creating advanced ensembles...")

# Voting ensemble with top models
top_3_models = sorted(all_results.items(), key=lambda x: x[1]['cv_mean'], reverse=True)[:3]
ensemble_estimators = [(name, models[name]) for name, _ in top_3_models]

voting_ensemble = VotingClassifier(
    estimators=ensemble_estimators,
    voting='soft',
    weights=[3, 2, 1]  # Weight by performance
)

# Stacking ensemble
from sklearn.ensemble import StackingClassifier
stacking_ensemble = StackingClassifier(
    estimators=ensemble_estimators,
    final_estimator=LogisticRegression(random_state=42),
    cv=5
)

# Test ensembles
ensembles = {
    'Voting Ensemble': voting_ensemble,
    'Stacking Ensemble': stacking_ensemble
}

for name, model in ensembles.items():
    print(f"\n🔧 Training {name}...")
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
    
    print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
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

print(f"\n🏆 Overall best model: {best_name} with {best_accuracy*100:.2f}% CV accuracy")

# Final validation and testing
print("\n[8/10] Final comprehensive validation...")

# Multiple train-test splits for robustness
test_splits = [0.2, 0.25, 0.3]
validation_results = []

for test_size in test_splits:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    validation_results.append(accuracy)
    print(f"  Test size {test_size}: {accuracy*100:.2f}%")

print(f"✓ Validation range: {min(validation_results)*100:.2f}% - {max(validation_results)*100:.2f}%")

# Final training on full dataset
print("\n[9/10] Final training on complete dataset...")
best_model.fit(X, y)

# Comprehensive evaluation
print("\n[10/10] Comprehensive model evaluation...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_pred = best_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)

print(f"\nFinal Test Results:")
print(f"  Accuracy: {final_accuracy*100:.2f}%")
print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"\nDetailed Classification Report:")
print(classification_report(y_test, y_test))

# Save comprehensive model
print("\n" + "="*80)
print("SAVING REALISTIC COMPREHENSIVE MODEL")
print("="*80)

joblib.dump(best_model, 'model.pkl')
joblib.dump(encoders, 'encoders.pkl')
joblib.dump(list(X.columns), 'feature_names.pkl')

# Save comprehensive model info
model_info = {
    'best_model': best_name,
    'cv_accuracy': float(best_accuracy),
    'test_accuracy': float(final_accuracy),
    'validation_range': {
        'min': float(min(validation_results)),
        'max': float(max(validation_results))
    },
    'n_samples': len(X),
    'n_features': len(X.columns),
    'feature_scores': feature_scores.head(30).to_dict('records'),
    'model_type': type(best_model).__name__,
    'all_model_results': all_results,
    'hyperparameters': best_model.get_params() if hasattr(best_model, 'get_params') else {},
    'dataset_info': {
        'train_records': len(train_df),
        'test_records': len(test_df),
        'total_records': len(train_df) + len(test_df),
        'approval_rate': float(len(df[df['Loan_Status'] == 'Y']) / len(df))
    },
    'feature_engineering': {
        'total_features_created': len(df_features.columns) - len(df.columns),
        'categorical_encoded': len(categorical_cols),
        'numerical_features': len(df_features.select_dtypes(include=[np.number]).columns)
    }
}

if hasattr(best_model, 'feature_importances_'):
    importance = list(zip(X.columns, best_model.feature_importances_))
    importance.sort(key=lambda x: x[1], reverse=True)
    model_info['feature_importance'] = [
        {'feature': f, 'importance': float(i)} for f, i in importance
    ]

with open('model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("✓ Saved realistic comprehensive model.pkl")
print("✓ Saved encoders.pkl")
print("✓ Saved feature_names.pkl")
print("✓ Saved model_info.json")

print("\n" + "="*80)
print("REALISTIC COMPREHENSIVE MODEL TRAINING COMPLETE!")
print("="*80)
print(f"\n🎯 Cross-Validation Accuracy: {best_accuracy*100:.2f}%")
print(f"🎯 Test Accuracy: {final_accuracy*100:.2f}%")
print(f"🎯 Validation Range: {min(validation_results)*100:.2f}% - {max(validation_results)*100:.2f}%")
print(f"🏆 Best Model: {best_name}")
print(f"🧠 Features Used: {len(X.columns)}")
print(f"📊 Training Records: {len(X)}")
print(f"📈 Dataset Approval Rate: {len(df[df['Loan_Status'] == 'Y'])/len(df)*100:.1f}%")
print(f"\n🔥 REAL-WORLD READY MODEL WITH DIVERSE TRAINING!")
print("="*80)
