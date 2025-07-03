from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)
CORS(app)

def clean_mood(mood):
    # Remove emojis and convert to lowercase
    return ''.join(char.lower() for char in mood if char.isalpha())

def prepare_training_data(df):
    # Create copies to avoid modifying original data
    df = df.copy()
    
    # Clean mood column
    df['mood'] = df['mood'].apply(clean_mood)
    
    # Convert gender to lowercase
    df['gender'] = df['gender'].str.lower()
    
    # Encode categorical variables
    global mood_encoder, gender_encoder
    mood_encoder = LabelEncoder()
    gender_encoder = LabelEncoder()
    
    df['gender'] = gender_encoder.fit_transform(df['gender'])
    df['mood'] = mood_encoder.fit_transform(df['mood'])
    
    # Prepare features and target
    X = df[['age', 'gender', 'blood_pressure', 'heart_rate', 'mood', 'cholesterol']]
    y = (df['mental_status'] == 'Negative').astype(int)
    
    return X, y

def train_model():
    # Load and prepare data
    df = pd.read_csv('mental_health_data.csv')
    X, y = prepare_training_data(df)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model

def validate_inputs(data):
    errors = []
    
    if not (0 <= data['age'] <= 120):
        errors.append("Age must be between 0 and 120")
    
    if not (40 <= data['heart_rate'] <= 200):
        errors.append("Heart rate must be between 40 and 200")
    
    if not (70 <= data['blood_pressure'] <= 200):
        errors.append("Blood pressure must be between 70 and 200")
    
    if not (100 <= data['cholesterol'] <= 600):
        errors.append("Cholesterol must be between 100 and 600")
    
    return errors

def assess_risk_factors(data, prediction):
    risk_factors = []
    severity = "low"
    
    if data['age'] < 30 and prediction == 1:
        risk_factors.append("Young age with concerning indicators")
        severity = "high"
    
    if data['heart_rate'] < 60:
        risk_factors.append("Low heart rate detected")
        severity = "moderate"
    elif data['heart_rate'] > 100:
        risk_factors.append("High heart rate detected")
        severity = "high"
    
    if data['blood_pressure'] < 90:
        risk_factors.append("Low blood pressure detected")
        severity = "high"
    elif data['blood_pressure'] > 140:
        risk_factors.append("High blood pressure detected")
        severity = "high"
    
    mood_str = clean_mood(data['mood'])
    if mood_str in ['sad', 'angry', 'isolated', 'anxious']:
        risk_factors.append(f"Negative mood state: {mood_str}")
        severity = "high"
    
    return risk_factors, severity

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Validate inputs
        errors = validate_inputs(data)
        if errors:
            return jsonify({'error': errors}), 400
        
        # Clean and prepare input data
        cleaned_mood = clean_mood(data['mood'])
        cleaned_gender = data['gender'].lower()
        
        # Create feature array
        features = [
            data['age'],
            gender_encoder.transform([cleaned_gender])[0],
            data['blood_pressure'],
            data['heart_rate'],
            mood_encoder.transform([cleaned_mood])[0],
            data['cholesterol']
        ]
        
        # Make prediction
        prediction = model.predict([features])[0]
        
        # Assess risk factors
        risk_factors, severity = assess_risk_factors(data, prediction)
        
        response = {
            'prediction': 'negative' if prediction == 1 else 'positive',
            'risk_factors': risk_factors,
            'severity': severity
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Train and save model
    model = train_model()
    
    app.run(debug=True, port=5000)