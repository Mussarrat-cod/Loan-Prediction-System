from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the trained model and encoders
try:
    print("Attempting to load model files...")
    model = joblib.load('model.pkl')
    encoders = joblib.load('encoders.pkl')
    feature_names = joblib.load('feature_names.pkl')
    print("✓ Model files loaded successfully!")
except Exception as e:
    print(f"Warning: Model files not found. Error: {e}")
    print("Please run train_and_save_model.py first")
    model = None
    encoders = {}
    feature_names = []

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    try:
        data = request.json
        
        # Extract and engineer features from frontend data
        applicant_income = int(data.get('ApplicantIncome', 50000))
        coapplicant_income = int(data.get('CoapplicantIncome', 0))
        loan_amount = int(data.get('LoanAmount', 10000))
        loan_term = int(data.get('Loan_Amount_Term', 360))
        credit_history = int(data.get('Credit_History', 1))
        married = 1 if data.get('Married', 'No') == 'Yes' else 0
        education = 0 if data.get('Education', 'Graduate') == 'Graduate' else 1
        property_area = 2 if data.get('Property_Area', 'Urban') == 'Urban' else (1 if data.get('Property_Area', 'Urban') == 'Semiurban' else 0)
        
        total_income = applicant_income + coapplicant_income
        
        features = {
            'Credit_History': credit_history,
            'Good_Credit_Flag': credit_history,
            'Credit_History_x_Income': credit_history * np.log1p(total_income),
            'Loan_to_Income_Ratio': loan_amount / (total_income + 1),
            'Total_Income_log': np.log1p(total_income),
            'LoanAmount_log': np.log1p(loan_amount),
            'Married': married,
            'Education': education,
            'Property_Area': property_area,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_term
        }
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # Ensure all required columns are present
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder columns to match training
        df = df[feature_names]
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0]
        confidence = max(probability) * 100
        
        # Reject if confidence is less than 60%
        if confidence < 60:
            prediction = 0
        # Override prediction if confidence > 80%
        elif confidence > 80:
            prediction = 1
        
        result = {
            'success': True,
            'prediction': int(prediction),
            'probability': float(probability[1]),
            'status': 'Approved' if prediction == 1 else 'Rejected',
            'confidence': f"{confidence:.2f}%"
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
