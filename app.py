from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model and preprocessing objects
model = joblib.load('model/best_rf_model.pkl')
scaler = joblib.load('model/scaler.pkl')
model_columns = joblib.load('model/model_columns.pkl')

# Define columns as used during training
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
multi_cat_cols = [
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaymentMethod'
]
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Define the preprocessing function
def preprocess_input(data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data])

    # Map 'Yes'/'No' to 1/0 for binary columns including SeniorCitizen
    yes_no_cols = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in yes_no_cols:
        input_df[col] = input_df[col].map({'Yes': 1, 'No': 0})

    # Encode 'gender' column
    if 'gender' in input_df.columns:
        input_df['gender'] = input_df['gender'].map({'Male': 1, 'Female': 0})
    else:
        input_df['gender'] = 0  # Default value if column is missing

    # Set default values for multi-category columns if not provided
    for col in multi_cat_cols:
        if col not in input_df.columns:
            input_df[col] = 'No'

    # Ensure numerical columns are numeric
    for col in num_cols:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
        else:
            input_df[col] = 0  # Default value if column is missing

    # Handle missing values after conversion
    input_df.fillna(0, inplace=True)

    # Perform One-Hot Encoding using get_dummies
    input_df = pd.get_dummies(input_df, columns=multi_cat_cols, drop_first=True)

    # Ensure that all expected columns are present in the input
    missing_cols = set(model_columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0  # Add missing columns with default value 0

    # Ensure the order of columns matches the training data
    input_df = input_df[model_columns]

    # Scale numerical features
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    return input_df

@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json(force=True)
        
        # Preprocess the input data
        processed_data = preprocess_input(data)

        # Make prediction
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data)

        # Return the prediction and confidence
        return jsonify({
            'Churn': int(prediction[0]),
            'Probability': float(np.max(probability))
        })

    except Exception as e:
        # Print the error message for debugging
        print(f"Error: {e}")
        # Return error message
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)