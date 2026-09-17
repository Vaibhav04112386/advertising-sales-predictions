
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model
model = joblib.load('linear_regression_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True) # Get data posted as JSON

    # Ensure input data matches expected features
    tv = data.get('TV')
    radio = data.get('Radio')
    newspaper = data.get('Newspaper')

    if None in [tv, radio, newspaper]:
        return jsonify({'error': 'Missing input features (TV, Radio, Newspaper)'}), 400

    # Create a DataFrame for prediction
    input_df = pd.DataFrame([[tv, radio, newspaper]], columns=['TV', 'Radio', 'Newspaper'])

    # Make prediction
    prediction = model.predict(input_df)[0]

    return jsonify({'predicted_sales': float(prediction)}) # Convert numpy float to native float for JSON serialization

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
