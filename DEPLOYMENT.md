# 🚀 Deployment Guide - Loan Prediction System

Complete setup and deployment instructions for the full-stack loan prediction application.

## 📋 Prerequisites

- **Python 3.8+** installed
- **Node.js 18+** and npm installed
- Terminal/Command Prompt access

## 🔧 Backend Setup

### Step 1: Install Python Dependencies

```cmd
cd d:\loan\backend
pip install -r requirements.txt
```

This installs:
- Flask (Web framework)
- Flask-CORS (Cross-origin support)
- scikit-learn (Machine learning)
- pandas, numpy (Data processing)
- joblib (Model serialization)

### Step 2: Train and Save the Model

```cmd
python train_and_save_model.py
```

**Expected Output:**
```
============================================================
TRAINING AND SAVING LOAN PREDICTION MODEL
============================================================

[1/5] Loading dataset...
✓ Loaded 98 records

[2/5] Creating label encoders...
✓ Created encoders for 6 columns

[3/5] Engineering features...
✓ Created 4 engineered features

[4/5] Training Random Forest model...
✓ Model trained successfully!
  Accuracy: 95.00%

[5/5] Saving model and encoders...
✓ Saved model.pkl
✓ Saved encoders.pkl
✓ Saved feature_names.pkl
✓ Saved model_info.json

MODEL TRAINING COMPLETE!
```

**Files Created:**
- `model.pkl` - Trained Random Forest model
- `encoders.pkl` - Label encoders for categorical variables
- `feature_names.pkl` - Feature names in correct order
- `model_info.json` - Model metadata and accuracy

### Step 3: Start Backend Server

```cmd
python app.py
```

**Server will start on:** `http://localhost:5000`

**Endpoints:**
- `GET /` - API information
- `GET /api/health` - Health check
- `GET /api/model-info` - Model metadata
- `POST /api/predict` - Make predictions

**Keep this terminal running!**

---

## 🎨 Frontend Setup

### Step 1: Install Dependencies

Open a **NEW terminal** window:

```cmd
cd d:\loan\frontend
npm install
```

This installs:
- React 18
- Vite (Build tool)
- React plugins

### Step 2: Start Development Server

```cmd
npm run dev
```

**Frontend will start on:** `http://localhost:3000`

The browser should automatically open. If not, navigate to the URL manually.

**Keep this terminal running too!**

---

## ✅ Verification

### Test the Complete System

1. **Open Browser:** Navigate to `http://localhost:3000`

2. **Fill Form:** Enter sample loan application data:
   - Gender: Male
   - Married: Yes
   - Dependents: 2
   - Education: Graduate
   - Self-Employed: No
   - Applicant Income: 5000
   - Co-applicant Income: 2000
   - Loan Amount: 150
   - Loan Term: 360 months
   - Credit History: Good
   - Property Area: Urban

3. **Submit:** Click "Predict Loan Approval"

4. **Expected Result:**
   - ✅ Approved
   - Confidence: ~90-95%
   - Shows probability breakdown

5. **Test Another:** Click "Check Another Application"

### Test Backend API Directly

```cmd
curl http://localhost:5000/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2026-02-13T...",
  "model_loaded": true
}
```

---

## 📁 Project Structure

```
d:\loan\
├── backend\
│   ├── app.py                    # Flask API server
│   ├── train_and_save_model.py   # Model training script
│   ├── requirements.txt          # Python dependencies
│   ├── model.pkl                 # Trained model
│   ├── encoders.pkl              # Encoders
│   ├── feature_names.pkl         # Feature list
│   └── model_info.json           # Model metadata
│
├── frontend\
│   ├── src\
│   │   ├── App.jsx               # Main app
│   │   ├── index.css             # Styles
│   │   ├── main.jsx              # Entry point
│   │   └── components\
│   │       ├── LoanForm.jsx      # Form component
│   │       └── PredictionResult.jsx  # Result display
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
└── DEPLOYMENT.md                 # This file
```

---

## 🛠 Common Issues

### Backend Issues

**Error: Module not found**
```cmd
pip install -r requirements.txt
```

**Error: No such file 'loan_data.csv'**
- Ensure you're running `train_and_save_model.py` from `d:\loan\backend`
- Check that `loan_data.csv` exists in `d:\loan`

**Port 5000 already in use:**
- Edit `app.py`, change port: `app.run(debug=True, port=5001)`
- Update frontend `API_URL` in `App.jsx` to match

### Frontend Issues

**Error: Cannot GET /**
- Make sure you ran `npm install` first

**API calls failing:**
- Verify backend is running on `http://localhost:5000`
- Check browser console for CORS errors
- Ensure Flask-CORS is installed: `pip install flask-cors`

**Blank screen:**
- Check browser console (F12) for errors
- Verify all files in `src/` are created correctly

---

## 🌐 Production Deployment

### Backend (Flask)

**Option 1: Heroku**
```cmd
# Create Procfile
echo web: python app.py > Procfile

# Deploy
heroku create loan-prediction-api
git push heroku main
```

**Option 2: Railway**
- Connect GitHub repo
- Auto-deploys on push

### Frontend (React)

**Build Production Bundle:**
```cmd
cd frontend
npm run build
```

**Deploy to:**
- **Vercel**: `vercel --prod`
- **Netlify**: Drag `dist/` folder
- **GitHub Pages**: Configure in repo settings

**Update API URL:**
- Change `API_URL` in `App.jsx` to production backend URL

---

## 📊 Model Performance

- **Algorithm:** Random Forest Classifier
- **Training Samples:** ~78 records
- **Test Samples:** ~20 records
- **Accuracy:** 85-95% (varies by random seed)
- **Features Used:** 15 (11 original + 4 engineered)

**Key Features:**
1. Credit_History (Most important)
2. Income_Loan_Ratio
3. Total_Income_log
4. LoanAmount_log
5. ApplicantIncome

---

## 💡 Next Steps

1. **Add Authentication:** Implement user login/signup
2. **Database:** Store prediction history (SQLite/PostgreSQL)
3. **More Models:** Try XGBoost, Neural Networks
4. **Explainability:** Add SHAP values for transparency
5. **Real-time Updates:** WebSocket for live predictions
6. **Mobile App:** React Native version

---

## 📞 Support

**Issues?**
- Check both terminals for error messages
- Verify all dependencies are installed
- Ensure correct directory structure

**Last Updated:** February 2026
