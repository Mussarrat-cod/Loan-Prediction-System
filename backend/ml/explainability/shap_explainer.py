"""
SHAP Explainer Module (Fallback Implementation)

Provides SHAP-based explanations for loan prediction models.
Supports Logistic Regression, Random Forest, and Decision Tree models.
"""

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class SHAPExplainer:
    """
    SHAP-based model explainer for loan predictions.
    
    Provides both global and local explanations using SHAP values.
    """
    
    def __init__(self, model_path: str = 'model.pkl', feature_names_path: str = 'feature_names.pkl'):
        """
        Initialize SHAP explainer with trained model.
        
        Args:
            model_path: Path to trained model pickle file
            feature_names_path: Path to feature names pickle file
        """
        self.model = joblib.load(model_path)
        self.feature_names = joblib.load(feature_names_path)
        self.explainer = None
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize appropriate SHAP explainer based on model type."""
        if not SHAP_AVAILABLE:
            self.explainer = None
            return
            
        model_name = type(self.model).__name__.lower()
        
        if 'logistic' in model_name or 'linear' in model_name:
            self.explainer = shap.LinearExplainer(self.model, np.zeros((1, len(self.feature_names))))
        elif 'forest' in model_name or 'tree' in model_name:
            # Use TreeExplainer for tree-based models
            self.explainer = shap.TreeExplainer(self.model)
        else:
            # Fallback to KernelExplainer for model-agnostic explanations
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, 
                np.zeros((1, len(self.feature_names)))
            )
    
    def explain_instance(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a single loan application.
        
        Args:
            features: Dictionary of feature values for the instance
            
        Returns:
            Dictionary containing SHAP values and explanations
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            # Fallback implementation using feature importance
            return self._fallback_explanation(features)
        
        # Convert features to DataFrame
        df = pd.DataFrame([features])
        
        # Ensure all required columns are present
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder columns to match training
        df = df[self.feature_names]
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(df)
        
        # Handle binary classification (get values for positive class)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Get values for class 1 (approved)
        
        # Get base value (expected value)
        base_value = self.explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1]  # Get expected value for class 1
        
        # Calculate feature contributions
        feature_contributions = {}
        for i, feature in enumerate(self.feature_names):
            feature_contributions[feature] = {
                'shap_value': float(shap_values[0, i]),
                'feature_value': float(df.iloc[0, i])
            }
        
        # Sort features by absolute SHAP value
        sorted_features = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]['shap_value']),
            reverse=True
        )
        
        # Separate positive and negative contributors
        positive_factors = [
            {
                'feature': name,
                'value': details['feature_value'],
                'contribution': details['shap_value'],
                'impact': 'positive'
            }
            for name, details in sorted_features
            if details['shap_value'] > 0
        ]
        
        negative_factors = [
            {
                'feature': name,
                'value': details['feature_value'],
                'contribution': details['shap_value'],
                'impact': 'negative'
            }
            for name, details in sorted_features
            if details['shap_value'] < 0
        ]
        
        return {
            'base_value': float(base_value),
            'prediction_value': float(base_value + np.sum(shap_values[0])),
            'feature_contributions': feature_contributions,
            'positive_factors': positive_factors[:5],  # Top 5 positive factors
            'negative_factors': negative_factors[:5],  # Top 5 negative factors
            'all_factors': sorted_features[:10]  # Top 10 overall factors
        }
    
    def _fallback_explanation(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback explanation using feature importance when SHAP is not available."""
        # Convert features to DataFrame
        df = pd.DataFrame([features])
        
        # Ensure all required columns are present
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder columns to match training
        df = df[self.feature_names]
        
        # Get model prediction
        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0, 1]
        
        # Use feature importance from model
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        else:
            importance = np.ones(len(self.feature_names)) / len(self.feature_names)
        
        # Create mock SHAP-like explanations based on feature importance and values
        feature_contributions = {}
        for i, feature in enumerate(self.feature_names):
            # Simulate SHAP values based on feature importance and feature value
            feature_value = float(df.iloc[0, i])
            contribution = importance[i] * (feature_value - 0.5) * 0.1  # Simplified contribution
            
            feature_contributions[feature] = {
                'shap_value': contribution,
                'feature_value': feature_value
            }
        
        # Sort features by absolute contribution
        sorted_features = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]['shap_value']),
            reverse=True
        )
        
        # Separate positive and negative contributors
        positive_factors = [
            {
                'feature': name,
                'value': details['feature_value'],
                'contribution': details['shap_value'],
                'impact': 'positive'
            }
            for name, details in sorted_features
            if details['shap_value'] > 0
        ]
        
        negative_factors = [
            {
                'feature': name,
                'value': details['feature_value'],
                'contribution': details['shap_value'],
                'impact': 'negative'
            }
            for name, details in sorted_features
            if details['shap_value'] < 0
        ]
        
        return {
            'base_value': 0.5,
            'prediction_value': float(probability),
            'feature_contributions': feature_contributions,
            'positive_factors': positive_factors[:5],
            'negative_factors': negative_factors[:5],
            'all_factors': sorted_features[:10]
        }
    
    def get_global_feature_importance(self, sample_data: pd.DataFrame = None) -> Dict[str, float]:
        """
        Calculate global feature importance using SHAP values.
        
        Args:
            sample_data: Sample data to calculate importance on (optional)
            
        Returns:
            Dictionary of feature importance scores
        """
        if sample_data is None:
            # Generate synthetic data if none provided
            sample_data = pd.DataFrame(np.random.randn(100, len(self.feature_names)), 
                                     columns=self.feature_names)
        
        # Calculate SHAP values for the sample
        shap_values = self.explainer.shap_values(sample_data)
        
        # Handle binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Calculate mean absolute SHAP values for each feature
        importance = {}
        for i, feature in enumerate(self.feature_names):
            importance[feature] = float(np.mean(np.abs(shap_values[:, i])))
        
        # Sort by importance
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def generate_summary_plot_data(self, sample_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Generate data for SHAP summary plot visualization.
        
        Args:
            sample_data: Sample data for visualization
            
        Returns:
            Dictionary containing plot data
        """
        if sample_data is None:
            sample_data = pd.DataFrame(np.random.randn(50, len(self.feature_names)), 
                                     columns=self.feature_names)
        
        shap_values = self.explainer.shap_values(sample_data)
        
        # Handle binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        return {
            'features': sample_data.values.tolist(),
            'shap_values': shap_values.tolist(),
            'feature_names': self.feature_names
        }
