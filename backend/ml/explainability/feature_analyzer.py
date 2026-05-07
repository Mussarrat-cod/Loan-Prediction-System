"""
Feature Analyzer Module

Provides comprehensive feature importance analysis and interpretation
for loan prediction models.
"""

import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


class FeatureAnalyzer:
    """
    Analyzes feature importance and provides insights for loan predictions.
    """
    
    def __init__(self, model_path: str = 'model.pkl', feature_names_path: str = 'feature_names.pkl'):
        """
        Initialize feature analyzer with trained model.
        
        Args:
            model_path: Path to trained model pickle file
            feature_names_path: Path to feature names pickle file
        """
        self.model = joblib.load(model_path)
        self.feature_names = joblib.load(feature_names_path)
        self.model_type = type(self.model).__name__
    
    def get_model_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance directly from the trained model.
        
        Returns:
            Dictionary of feature importance scores
        """
        importance = {}
        
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models (Random Forest, Decision Tree)
            for i, feature in enumerate(self.feature_names):
                importance[feature] = float(self.model.feature_importances_[i])
        
        elif hasattr(self.model, 'coef_'):
            # Linear models (Logistic Regression)
            coef = self.model.coef_[0] if len(self.model.coef_.shape) > 1 else self.model.coef_
            for i, feature in enumerate(self.feature_names):
                importance[feature] = float(abs(coef[i]))
        
        else:
            # Model doesn't have built-in importance
            importance = {feature: 0.0 for feature in self.feature_names}
        
        # Sort by importance
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    def analyze_feature_impact(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze how each feature impacts the prediction for a specific instance.
        
        Args:
            features: Dictionary of feature values for the instance
            
        Returns:
            Dictionary containing feature impact analysis
        """
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Ensure all required columns are present
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder columns to match training
        df = df[self.feature_names]
        
        # Get base prediction
        base_prediction = self.model.predict_proba(df)[0, 1]
        
        # Analyze impact of each feature by setting it to its mean value
        feature_impacts = {}
        
        # Calculate mean values (using reasonable defaults for loan features)
        mean_values = {
            'Credit_History': 0.8,
            'Good_Credit_Flag': 0.8,
            'Credit_History_x_Income': 10.0,
            'Loan_to_Income_Ratio': 0.3,
            'Total_Income_log': 10.0,
            'LoanAmount_log': 9.0,
            'Married': 0.6,
            'Education': 0.2,
            'Property_Area': 1.0,
            'ApplicantIncome': 5000,
            'CoapplicantIncome': 2000,
            'LoanAmount': 150,
            'Loan_Amount_Term': 360
        }
        
        for feature in self.feature_names:
            # Create modified data with feature set to mean
            modified_df = df.copy()
            modified_df[feature] = mean_values.get(feature, 0)
            
            # Get prediction with modified feature
            modified_prediction = self.model.predict_proba(modified_df)[0, 1]
            
            # Calculate impact
            impact = modified_prediction - base_prediction
            feature_impacts[feature] = {
                'original_value': float(df[feature].iloc[0]),
                'impact_score': float(impact),
                'impact_direction': 'positive' if impact > 0 else 'negative' if impact < 0 else 'neutral'
            }
        
        # Sort by absolute impact
        sorted_impacts = sorted(
            feature_impacts.items(),
            key=lambda x: abs(x[1]['impact_score']),
            reverse=True
        )
        
        return {
            'base_prediction': float(base_prediction),
            'feature_impacts': dict(sorted_impacts),
            'top_positive_factors': [
                {'feature': name, **details}
                for name, details in sorted_impacts
                if details['impact_direction'] == 'positive'
            ][:5],
            'top_negative_factors': [
                {'feature': name, **details}
                for name, details in sorted_impacts
                if details['impact_direction'] == 'negative'
            ][:5]
        }
    
    def get_feature_statistics(self, data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Get comprehensive statistics for each feature.
        
        Args:
            data: Dataset to calculate statistics from (optional)
            
        Returns:
            Dictionary containing feature statistics
        """
        if data is None:
            # Generate synthetic data for demonstration
            np.random.seed(42)
            data = pd.DataFrame({
                'Credit_History': np.random.binomial(1, 0.8, 100),
                'Good_Credit_Flag': np.random.binomial(1, 0.8, 100),
                'Credit_History_x_Income': np.random.exponential(10, 100),
                'Loan_to_Income_Ratio': np.random.exponential(0.3, 100),
                'Total_Income_log': np.random.normal(10, 1, 100),
                'LoanAmount_log': np.random.normal(9, 0.5, 100),
                'Married': np.random.binomial(1, 0.6, 100),
                'Education': np.random.binomial(1, 0.2, 100),
                'Property_Area': np.random.randint(0, 3, 100),
                'ApplicantIncome': np.random.exponential(5000, 100),
                'CoapplicantIncome': np.random.exponential(2000, 100),
                'LoanAmount': np.random.exponential(150, 100),
                'Loan_Amount_Term': np.random.choice([180, 240, 360], 100)
            })
        
        stats = {}
        for feature in self.feature_names:
            if feature in data.columns:
                feature_data = data[feature]
                stats[feature] = {
                    'mean': float(feature_data.mean()),
                    'median': float(feature_data.median()),
                    'std': float(feature_data.std()),
                    'min': float(feature_data.min()),
                    'max': float(feature_data.max()),
                    'q25': float(feature_data.quantile(0.25)),
                    'q75': float(feature_data.quantile(0.75)),
                    'missing_rate': float(feature_data.isnull().mean() * 100)
                }
        
        return stats
    
    def get_feature_correlations(self, data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Calculate feature correlations with the target variable.
        
        Args:
            data: Dataset to calculate correlations from (optional)
            
        Returns:
            Dictionary containing feature correlations
        """
        if data is None:
            # Generate synthetic data with target
            np.random.seed(42)
            n_samples = 100
            data = pd.DataFrame({
                'Credit_History': np.random.binomial(1, 0.8, n_samples),
                'Good_Credit_Flag': np.random.binomial(1, 0.8, n_samples),
                'Credit_History_x_Income': np.random.exponential(10, n_samples),
                'Loan_to_Income_Ratio': np.random.exponential(0.3, n_samples),
                'Total_Income_log': np.random.normal(10, 1, n_samples),
                'LoanAmount_log': np.random.normal(9, 0.5, n_samples),
                'Married': np.random.binomial(1, 0.6, n_samples),
                'Education': np.random.binomial(1, 0.2, n_samples),
                'Property_Area': np.random.randint(0, 3, n_samples),
                'ApplicantIncome': np.random.exponential(5000, n_samples),
                'CoapplicantIncome': np.random.exponential(2000, n_samples),
                'LoanAmount': np.random.exponential(150, n_samples),
                'Loan_Amount_Term': np.random.choice([180, 240, 360], n_samples)
            })
            
            # Create synthetic target based on key features
            data['Loan_Status'] = (
                (data['Credit_History'] > 0.5) & 
                (data['Loan_to_Income_Ratio'] < 0.5) & 
                (data['Total_Income_log'] > 9)
            ).astype(int)
        
        # Calculate correlations with target
        correlations = {}
        if 'Loan_Status' in data.columns:
            for feature in self.feature_names:
                if feature in data.columns:
                    corr = data[feature].corr(data['Loan_Status'])
                    correlations[feature] = float(corr) if not np.isnan(corr) else 0.0
        
        # Sort by absolute correlation
        return dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))
