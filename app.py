from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the saved model and scaler
model = joblib.load('obesity_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        input_data = {
            'Gender': request.form['Gender'],
            'Age': float(request.form['Age']),
            'Height': float(request.form['Height']),
            'Weight': float(request.form['Weight']),
            'family_history_with_overweight': request.form['family_history_with_overweight'],
            'FAVC': request.form['FAVC'],
            'FCVC': float(request.form['FCVC']),
            'NCP': float(request.form['NCP']),
            'CAEC': request.form['CAEC'],
            'SMOKE': request.form['SMOKE'],
            'CH2O': float(request.form['CH2O']),
            'SCC': request.form['SCC'],
            'FAF': float(request.form['FAF']),
            'TUE': float(request.form['TUE']),
            'CALC': request.form['CALC'],
            'MTRANS': request.form['MTRANS']
        }

        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # One-hot encoding same as training
        cat_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
        df = pd.get_dummies(df, columns=cat_features)

        # Reindex to match training features
        with open('features.txt', 'r') as f:
            all_features = f.read().splitlines()
        df = df.reindex(columns=all_features, fill_value=0)

        # Add BMI
        df['BMI'] = df['Weight'] / (df['Height'] ** 2)

        # Scale
        X = scaler.transform(df)

        # Predict
        prediction = model.predict(X)
        result = label_encoder.inverse_transform(prediction)[0]

        return render_template('index.html', prediction=result)

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
