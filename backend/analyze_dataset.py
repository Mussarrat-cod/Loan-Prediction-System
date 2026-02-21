import pandas as pd
import numpy as np

# Load datasets
train_df = pd.read_csv('../train.csv')
test_df = pd.read_csv('../test.csv')

print('='*80)
print('COMPREHENSIVE DATASET ATTRIBUTE ANALYSIS')
print('='*80)

print(f'\n📊 DATASET OVERVIEW:')
print(f'  Training Records: {len(train_df):,}')
print(f'  Test Records: {len(test_df):,}')
print(f'  Total Records: {len(train_df) + len(test_df):,}')
print(f'  Features: {len(train_df.columns)}')

print(f'\n📋 ATTRIBUTES & COLUMNS:')
for i, col in enumerate(train_df.columns, 1):
    dtype = train_df[col].dtype
    missing = train_df[col].isnull().sum()
    unique_count = train_df[col].nunique()
    
    if dtype == 'object':
        unique_vals = train_df[col].unique()[:5]
        print(f'  {i:2d}. {col:25} (Categorical)')
        print(f'     Type: {str(dtype):10} | Missing: {missing:3d} | Unique: {unique_count:3d}')
        print(f'     Values: {list(unique_vals)}')
    else:
        min_val = train_df[col].min()
        max_val = train_df[col].max()
        mean_val = train_df[col].mean()
        median_val = train_df[col].median()
        
        print(f'  {i:2d}. {col:25} (Numerical)')
        print(f'     Type: {str(dtype):10} | Missing: {missing:3d} | Unique: {unique_count:3d}')
        print(f'     Range: {min_val:,.0f} - {max_val:,.0f}')
        print(f'     Mean: {mean_val:,.2f} | Median: {median_val:,.2f}')

print(f'\n🎯 TARGET VARIABLE (Loan_Status):')
loan_status_counts = train_df['Loan_Status'].value_counts()
print(f'  Approved (Y): {loan_status_counts.get("Y", 0):,} ({loan_status_counts.get("Y", 0)/len(train_df)*100:.1f}%)')
print(f'  Rejected (N): {loan_status_counts.get("N", 0):,} ({loan_status_counts.get("N", 0)/len(train_df)*100:.1f}%)')

print(f'\n📈 KEY INSIGHTS:')
print(f'  Approval Rate: {loan_status_counts.get("Y", 0)/len(train_df)*100:.1f}%')
print(f'  Data Quality: {((len(train_df) - train_df.isnull().sum().sum())/ (len(train_df) * len(train_df.columns)) * 100):.1f}% complete')

print(f'\n🔍 CORRELATION WITH TARGET:')
# Calculate correlation for numerical columns
numerical_cols = train_df.select_dtypes(include=[np.number]).columns
if len(numerical_cols) > 0:
    correlations = train_df[numerical_cols].corrwith(train_df['Loan_Status'].map({'Y': 1, 'N': 0}))
    top_corr = correlations.abs().sort_values(ascending=False).head(5)
    for col, corr in top_corr.items():
        print(f'  {col:20}: {corr:.4f}')

print(f'\n👥 DEMOGRAPHIC BREAKDOWN:')
print('  Gender Distribution:')
gender_counts = train_df['Gender'].value_counts()
for gender, count in gender_counts.items():
    print(f'    {gender}: {count} ({count/len(train_df)*100:.1f}%)')

print('\n  Marital Status:')
married_counts = train_df['Married'].value_counts()
for status, count in married_counts.items():
    print(f'    {status}: {count} ({count/len(train_df)*100:.1f}%)')

print('\n  Education Level:')
edu_counts = train_df['Education'].value_counts()
for edu, count in edu_counts.items():
    print(f'    {edu}: {count} ({count/len(train_df)*100:.1f}%)')

print('\n  Property Area:')
property_counts = train_df['Property_Area'].value_counts()
for area, count in property_counts.items():
    print(f'    {area}: {count} ({count/len(train_df)*100:.1f}%)')

print(f'\n💰 FINANCIAL BREAKDOWN:')
print(f'  Applicant Income:')
print(f'    Mean: ${train_df["ApplicantIncome"].mean():,.0f}')
print(f'    Median: ${train_df["ApplicantIncome"].median():,.0f}')
print(f'    Range: ${train_df["ApplicantIncome"].min():,.0f} - ${train_df["ApplicantIncome"].max():,.0f}')

print(f'\n  Co-applicant Income:')
print(f'    Mean: ${train_df["CoapplicantIncome"].mean():,.0f}')
print(f'    Median: ${train_df["CoapplicantIncome"].median():,.0f}')
print(f'    Zero Income: {(train_df["CoapplicantIncome"] == 0).sum()} ({(train_df["CoapplicantIncome"] == 0).sum()/len(train_df)*100:.1f}%)')

print(f'\n  Loan Amount:')
print(f'    Mean: ${train_df["LoanAmount"].mean():,.0f}')
print(f'    Median: ${train_df["LoanAmount"].median():,.0f}')
print(f'    Range: ${train_df["LoanAmount"].min():,.0f} - ${train_df["LoanAmount"].max():,.0f}')

print(f'\n  Credit History:')
credit_counts = train_df['Credit_History'].value_counts()
for credit, count in credit_counts.items():
    credit_label = 'Good Credit' if credit == 1 else 'Poor Credit'
    print(f'    {credit_label}: {count} ({count/len(train_df)*100:.1f}%)')

print(f'\n' + '='*80)
print('DATASET ANALYSIS COMPLETE!')
print('='*80)
