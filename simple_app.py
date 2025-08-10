# Simple Kidney Disease Detection App
# Easy to understand and use - perfect for beginners!

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

app = Flask(__name__)

# Global variables to store our model and scaler
model = None
scaler = None

def load_or_create_model():
    """Load existing model or create a new one"""
    global model, scaler
    
    # Try to load existing model
    if os.path.exists('simple_model.pkl') and os.path.exists('simple_scaler.pkl'):
        try:
            model = joblib.load('simple_model.pkl')
            scaler = joblib.load('simple_scaler.pkl')
            print("✅ Loaded existing model!")
            return True
        except:
            print("❌ Could not load existing model, creating new one...")
    
    # Create new model if none exists
    print("🔧 Creating new model...")
    
    # Load and prepare data
    df = pd.read_csv('../kidney_disease.csv')
    
    # Simple preprocessing - just the most important features
    important_features = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'hemo', 'pcv']
    
    # Clean the data
    for col in important_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
    
    # Prepare features and target
    X = df[important_features]
    y = (df['classification'] == 'ckd').astype(int)  # 1 for disease, 0 for no disease
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train a simple Random Forest model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_scaled, y)
    
    # Save the model and scaler
    joblib.dump(model, 'simple_model.pkl')
    joblib.dump(scaler, 'simple_scaler.pkl')
    
    print("✅ Model created and saved!")
    return True

def predict_kidney_disease(data):
    """Make prediction using our model"""
    try:
        # Scale the input data
        data_scaled = scaler.transform([data])
        
        # Make prediction
        prediction = model.predict(data_scaled)[0]
        probability = model.predict_proba(data_scaled)[0]
        
        # Get confidence
        confidence = max(probability) * 100
        
        if prediction == 1:
            result = "Chronic Kidney Disease Detected"
            risk_level = "High" if confidence > 80 else "Medium" if confidence > 60 else "Low"
        else:
            result = "No Kidney Disease Detected"
            risk_level = "Low"
        
        return result, confidence, risk_level
        
    except Exception as e:
        return f"Error: {str(e)}", 0, "Unknown"

# Load model when app starts
print("🚀 Starting Kidney Disease Detection App...")
load_or_create_model()

@app.route('/')
def home():
    """Home page - simple and clean"""
    return render_template('simple_home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page - handle form submission"""
    if request.method == 'POST':
        try:
            # Get form data
            age = float(request.form['age'])
            blood_pressure = float(request.form['blood_pressure'])
            specific_gravity = float(request.form['specific_gravity'])
            albumin = float(request.form['albumin'])
            sugar = float(request.form['sugar'])
            blood_glucose = float(request.form['blood_glucose'])
            blood_urea = float(request.form['blood_urea'])
            serum_creatinine = float(request.form['serum_creatinine'])
            hemoglobin = float(request.form['hemoglobin'])
            packed_cell_volume = float(request.form['packed_cell_volume'])
            
            # Create feature array
            features = [
                age, blood_pressure, specific_gravity, albumin, sugar,
                blood_glucose, blood_urea, serum_creatinine, hemoglobin, packed_cell_volume
            ]
            
            # Make prediction
            result, confidence, risk_level = predict_kidney_disease(features)
            
            return render_template('simple_result.html', 
                                result=result, 
                                confidence=f"{confidence:.1f}%",
                                risk_level=risk_level,
                                features=features)
            
        except Exception as e:
            return render_template('simple_result.html', 
                                result=f"Error: {str(e)}", 
                                confidence="0%",
                                risk_level="Unknown",
                                features=[])
    
    return render_template('simple_predict.html')

if __name__ == '__main__':
    print("🌐 Web app starting...")
    print("📱 Open your browser and go to: http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000) 