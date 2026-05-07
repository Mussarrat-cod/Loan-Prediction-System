"""
Explanation Generator Module

Generates human-readable explanations for loan prediction decisions.
Provides clear, business-friendly explanations for both technical and non-technical users.
"""

import numpy as np
from typing import Dict, List, Any, Optional
import datetime


class ExplanationGenerator:
    """
    Generates human-readable explanations for loan predictions.
    """
    
    def __init__(self):
        """Initialize the explanation generator."""
        self.feature_descriptions = {
            'Credit_History': 'Credit History Score',
            'Good_Credit_Flag': 'Good Credit Standing',
            'Credit_History_x_Income': 'Credit-Income Interaction',
            'Loan_to_Income_Ratio': 'Loan to Income Ratio',
            'Total_Income_log': 'Total Household Income',
            'LoanAmount_log': 'Loan Amount',
            'Married': 'Marital Status',
            'Education': 'Education Level',
            'Property_Area': 'Property Location',
            'ApplicantIncome': 'Applicant Income',
            'CoapplicantIncome': 'Co-applicant Income',
            'LoanAmount': 'Requested Loan Amount',
            'Loan_Amount_Term': 'Loan Term'
        }
        
        self.positive_templates = [
            "Your {feature} of {value} strengthens your application.",
            "Having {feature} at {value} positively impacts your eligibility.",
            "The {feature} of {value} works in your favor.",
            "Your {feature} being {value} increases your approval chances."
        ]
        
        self.negative_templates = [
            "Your {feature} of {value} raises concerns.",
            "The {feature} of {value} negatively affects your application.",
            "Having {feature} at {value} reduces your approval likelihood.",
            "Your {feature} being {value} works against your application."
        ]
    
    def generate_explanation(self, prediction_result: Dict[str, Any], 
                           shap_explanation: Dict[str, Any],
                           feature_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive human-readable explanation.
        
        Args:
            prediction_result: Model prediction results
            shap_explanation: SHAP-based explanation
            feature_analysis: Feature impact analysis
            
        Returns:
            Dictionary containing human-readable explanations
        """
        prediction = prediction_result.get('prediction', 0)
        confidence = prediction_result.get('confidence', 0)
        
        # Generate main decision explanation
        decision_explanation = self._generate_decision_explanation(
            prediction, confidence, shap_explanation, feature_analysis
        )
        
        # Generate factor explanations
        positive_factors = self._generate_factor_explanations(
            shap_explanation.get('positive_factors', []),
            positive=True
        )
        
        negative_factors = self._generate_factor_explanations(
            shap_explanation.get('negative_factors', []),
            positive=False
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            prediction, negative_factors, feature_analysis
        )
        
        # Generate risk assessment
        risk_assessment = self._generate_risk_assessment(
            prediction, confidence, shap_explanation
        )
        
        return {
            'decision_explanation': decision_explanation,
            'positive_factors': positive_factors,
            'negative_factors': negative_factors,
            'recommendations': recommendations,
            'risk_assessment': risk_assessment,
            'summary': self._generate_summary(prediction, confidence, positive_factors, negative_factors)
        }
    
    def _generate_decision_explanation(self, prediction: int, confidence: float,
                                    shap_explanation: Dict[str, Any],
                                    feature_analysis: Dict[str, Any]) -> str:
        """Generate the main decision explanation."""
        if prediction == 1:  # Approved
            if confidence >= 80:
                return ("Based on our analysis, your loan application has been **approved** with high confidence. "
                       f"Your profile shows strong indicators of loan repayment capability with {confidence:.1f}% confidence.")
            elif confidence >= 60:
                return ("Your loan application has been **approved**. While there are some areas that could be stronger, "
                       f"your overall profile meets our approval criteria with {confidence:.1f}% confidence.")
            else:
                return ("Your loan application has been **approved** after careful consideration. "
                       "While your confidence score is moderate, positive factors in your application outweigh the concerns.")
        else:  # Rejected
            if confidence >= 80:
                return ("Unfortunately, your loan application has been **rejected** with high confidence. "
                       f"Significant concerns were identified that indicate high risk with {confidence:.1f}% confidence.")
            elif confidence >= 60:
                return ("Your loan application has been **rejected**. While there are some positive aspects, "
                       f"the identified risks outweigh the benefits with {confidence:.1f}% confidence.")
            else:
                return ("Your loan application has been **rejected**. The decision was borderline, but "
                       "the identified concerns suggest it would be prudent to decline at this time.")
    
    def _generate_factor_explanations(self, factors: List[Dict[str, Any]], 
                                    positive: bool) -> List[Dict[str, Any]]:
        """Generate explanations for individual factors."""
        explanations = []
        templates = self.positive_templates if positive else self.negative_templates
        
        for factor in factors[:5]:  # Limit to top 5 factors
            feature = factor.get('feature', '')
            value = factor.get('value', 0)
            contribution = factor.get('contribution', 0)
            
            # Format value for display
            formatted_value = self._format_feature_value(feature, value)
            
            # Generate explanation
            import random
            template = random.choice(templates)
            explanation = template.format(
                feature=self.feature_descriptions.get(feature, feature),
                value=formatted_value
            )
            
            explanations.append({
                'feature': feature,
                'feature_name': self.feature_descriptions.get(feature, feature),
                'value': formatted_value,
                'contribution': float(contribution),
                'explanation': explanation,
                'impact_strength': self._assess_impact_strength(abs(contribution))
            })
        
        return explanations
    
    def _generate_recommendations(self, prediction: int, negative_factors: List[Dict[str, Any]],
                               feature_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if prediction == 0:  # Rejected
            recommendations.append("Consider improving your credit history before reapplying.")
            
            # Check specific factors for targeted recommendations
            for factor in negative_factors[:3]:
                feature = factor.get('feature', '')
                value = factor.get('value', 0)
                
                if 'Credit_History' in feature and value < 0.5:
                    recommendations.append("Work on building a stronger credit history by paying bills on time.")
                elif 'Loan_to_Income_Ratio' in feature and value > 0.4:
                    recommendations.append("Consider applying for a smaller loan amount or increasing your income.")
                elif 'Total_Income_log' in feature and value < 9:
                    recommendations.append("Consider adding a co-applicant with stable income to strengthen your application.")
                elif 'Education' in feature and value > 0.5:
                    recommendations.append("Highlight professional certifications and work experience to compensate for education level.")
        
        else:  # Approved
            recommendations.append("Maintain your current credit standing to ensure future loan success.")
            if len(negative_factors) > 2:
                recommendations.append("Address the identified concerns to improve your loan terms in the future.")
        
        return recommendations
    
    def _generate_risk_assessment(self, prediction: int, confidence: float,
                                shap_explanation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk assessment information."""
        base_value = shap_explanation.get('base_value', 0.5)
        prediction_value = shap_explanation.get('prediction_value', 0.5)
        
        risk_level = 'Low' if prediction == 1 and confidence >= 70 else 'Medium' if confidence >= 60 else 'High'
        
        return {
            'risk_level': risk_level,
            'base_probability': f"{base_value * 100:.1f}%",
            'adjusted_probability': f"{prediction_value * 100:.1f}%",
            'confidence_score': f"{confidence:.1f}%",
            'risk_factors_count': len(shap_explanation.get('negative_factors', [])),
            'strength_factors_count': len(shap_explanation.get('positive_factors', []))
        }
    
    def _generate_summary(self, prediction: int, confidence: float,
                         positive_factors: List[Dict[str, Any]],
                         negative_factors: List[Dict[str, Any]]) -> str:
        """Generate a concise summary of the decision."""
        status = "Approved" if prediction == 1 else "Rejected"
        
        if prediction == 1:
            if len(positive_factors) > len(negative_factors):
                return (f"Loan {status}: Your application shows {len(positive_factors)} strong positive factors "
                       f"compared to {len(negative_factors)} concerns, leading to approval with {confidence:.1f}% confidence.")
            else:
                return (f"Loan {status}: Despite some concerns, your key strengths outweigh the risks, "
                       f"resulting in approval with {confidence:.1f}% confidence.")
        else:
            return (f"Loan {status}: Your application shows {len(negative_factors)} significant concerns "
                   f"compared to {len(positive_factors)} strengths, leading to rejection with {confidence:.1f}% confidence.")
    
    def _format_feature_value(self, feature: str, value: float) -> str:
        """Format feature values for human-readable display."""
        if 'Credit_History' in feature or 'Good_Credit_Flag' in feature:
            return "Good" if value > 0.5 else "Poor"
        elif 'Married' in feature:
            return "Married" if value > 0.5 else "Single"
        elif 'Education' in feature:
            return "Not Graduate" if value > 0.5 else "Graduate"
        elif 'Property_Area' in feature:
            areas = {0: "Rural", 1: "Semiurban", 2: "Urban"}
            return areas.get(int(value), "Unknown")
        elif 'Loan_to_Income_Ratio' in feature:
            return f"{value:.2%}"
        elif '_log' in feature:
            return f"${np.exp(value):,.0f}"
        elif 'Income' in feature:
            return f"${value:,.0f}"
        elif 'LoanAmount' in feature:
            return f"${value:,.0f}"
        elif 'Loan_Amount_Term' in feature:
            return f"{int(value)} months"
        else:
            return str(value)
    
    def _assess_impact_strength(self, contribution: float) -> str:
        """Assess the strength of feature contribution."""
        if contribution >= 0.5:
            return "Very High"
        elif contribution >= 0.2:
            return "High"
        elif contribution >= 0.1:
            return "Medium"
        elif contribution >= 0.05:
            return "Low"
        else:
            return "Very Low"
