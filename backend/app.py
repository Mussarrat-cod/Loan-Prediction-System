from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
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
        
        result = {
            'success': True,
            'prediction': int(prediction),
            'probability': float(probability[1]),
            'status': 'Approved' if prediction == 1 else 'Rejected',
            'confidence': f"{max(probability)*100:.2f}%"
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    user_message_lower = user_message.lower()
    
    if not user_message:
        return jsonify({
            'success': False,
            'error': 'Message is required'
        }), 400
    
    ALPHA_VANTAGE_KEY = "YOUR_ALPHA_VANTAGE_KEY"
    HUGGINGFACE_TOKEN = "YOUR_HUGGINGFACE_TOKEN"
    reply = ""
    
    # Check if requesting market data (use Alpha Vantage)
    is_market_data = any(keyword in user_message_lower for keyword in [
        'price of', 'stock', 'exchange', 'forex', 'crypto', 'bitcoin', 'ethereum', 
        'btc', 'eth', 'aapl', 'googl', 'tsla', 'msft', 'amzn'
    ])
    
    if is_market_data:
        # ALPHA VANTAGE - Stock Price
        symbol = None
        if 'aapl' in user_message_lower:
            symbol = 'AAPL'
        elif 'googl' in user_message_lower or 'google' in user_message_lower:
            symbol = 'GOOGL'
        elif 'tsla' in user_message_lower or 'tesla' in user_message_lower:
            symbol = 'TSLA'
        elif 'msft' in user_message_lower or 'microsoft' in user_message_lower:
            symbol = 'MSFT'
        elif 'amzn' in user_message_lower or 'amazon' in user_message_lower:
            symbol = 'AMZN'
        elif 'bitcoin' in user_message_lower or 'btc' in user_message_lower:
            symbol = 'BTC'
        elif 'ethereum' in user_message_lower or 'eth' in user_message_lower:
            symbol = 'ETH'
        
        if symbol:
            try:
                if symbol in ['BTC', 'ETH']:
                    # Crypto
                    url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={symbol}&to_currency=USD&apikey={ALPHA_VANTAGE_KEY}"
                else:
                    # Stock
                    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_KEY}"
                
                response = requests.get(url)
                data = response.json()
                
                if symbol in ['BTC', 'ETH']:
                    price = data.get('Realtime Currency Exchange Rate', {}).get('5. Exchange Rate', 'N/A')
                    reply = f"Current {symbol} price: ${price}"
                else:
                    price = data.get('Global Quote', {}).get('05. price', 'N/A')
                    change = data.get('Global Quote', {}).get('10. change percent', 'N/A')
                    reply = f"{symbol} is trading at ${price} ({change} today)"
                    
            except Exception as e:
                reply = f"Sorry, I couldn't fetch the price data. Error: {str(e)}"
        else:
            reply = "I can help you get prices for stocks like AAPL, GOOGL, TSLA, MSFT, AMZN or cryptocurrencies like BTC, ETH. Just ask!"
    else:
        # HUGGINGFACE - General conversation
        try:
            API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
            headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
            
            payload = {"inputs": user_message}
            response = requests.post(API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    reply = result[0].get('generated_text', user_message)
                else:
                    reply = "I'm here to help with loan predictions and financial questions!"
            else:
                reply = "I'm your financial assistant. I can help with loan predictions and provide basic financial advice."
                
        except Exception as e:
            reply = "I'm your financial assistant. Ask me about loan eligibility, credit scores, or request stock/crypto prices!"
    
    return jsonify({
        'success': True,
        'reply': reply
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
