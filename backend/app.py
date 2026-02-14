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
    model = joblib.load('model.pkl')
    encoders = joblib.load('encoders.pkl')
    feature_names = joblib.load('feature_names.pkl')
except:
    print("Warning: Model files not found. Please run train_and_save_model.py first")
    model = None
    encoders = {}
    feature_names = []

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    try:
        data = request.json
        
        # Extract features from request
        features = {
            'person_age': int(data.get('age', 30)),
            'person_income': int(data.get('income', 50000)),
            'person_home_ownership': data.get('home_ownership', 'RENT'),
            'person_emp_length': float(data.get('emp_length', 5.0)),
            'loan_intent': data.get('loan_intent', 'EDUCATION'),
            'loan_grade': data.get('loan_grade', 'A'),
            'loan_amnt': int(data.get('loan_amount', 10000)),
            'loan_int_rate': float(data.get('interest_rate', 10.5)),
            'loan_percent_income': float(data.get('loan_percent_income', 0.2)),
            'cb_person_default_on_file': data.get('default_on_file', 'N'),
            'cb_person_cred_hist_length': int(data.get('credit_history_length', 5))
        }
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # Encode categorical variables
        for column in ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']:
            if column in encoders:
                df[column] = encoders[column].transform(df[column])
        
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

@app.route('/chat', methods=['POST'])
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

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
