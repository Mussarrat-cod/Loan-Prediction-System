"""
Explainability Utilities

Common utility functions for XAI functionality.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import joblib


class ExplainabilityUtils:
    """Utility functions for explainability module."""
    
    @staticmethod
    def preprocess_input_data(data: Dict[str, Any], feature_names: List[str]) -> pd.DataFrame:
        """Preprocess input data for model prediction."""
        df = pd.DataFrame([data])
        
        # Ensure all required columns are present
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder columns to match training
        df = df[feature_names]
        return df
    
    @staticmethod
    def calculate_confidence_score(probabilities: np.ndarray) -> float:
        """Calculate confidence score from prediction probabilities."""
        return float(np.max(probabilities) * 100)
    
    @staticmethod
    def format_currency(value: float) -> str:
        """Format value as currency."""
        return f"${value:,.0f}"
    
    @staticmethod
    def format_percentage(value: float) -> str:
        """Format value as percentage."""
        return f"{value:.1f}%"
