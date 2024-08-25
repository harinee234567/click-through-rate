from flask import Flask, request, jsonify
import pandas as pd
import xgboost as xgb
import os

app = Flask(__name__)

# Path to the model file
model_path = 'xgb_model.bin'
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found. Please check the file path.")

try:
    model = xgb.Booster()
    model.load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)
    
    # Convert the data into a DataFrame or appropriate format for your model
    input_data = pd.DataFrame([data])
    
    # Process Timestamp if needed
    if 'Timestamp' in input_data.columns:
        input_data['Timestamp'] = pd.to_datetime(input_data['Timestamp'])
        input_data['Year'] = input_data['Timestamp'].dt.year
        input_data['Month'] = input_data['Timestamp'].dt.month
        input_data['Day'] = input_data['Timestamp'].dt.day
        input_data['Hour'] = input_data['Timestamp'].dt.hour
    
    # Handle missing features in data
    features = ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage',
                'Ad Topic Line', 'City', 'Gender', 'Country', 'Year', 'Month', 'Day', 'Hour']
    
    for feature in features:
        if feature not in input_data.columns:
            input_data[feature] = 0  # Default value; adjust as needed

    # Ensure the order of columns matches the model's expected feature order
    input_data = input_data[features]

    # Convert categorical features to numerical
    categorical_features = ['Ad Topic Line', 'City', 'Gender', 'Country']
    for feature in categorical_features:
        if feature in input_data.columns:
            input_data[feature] = input_data[feature].astype('category').cat.codes

    # Convert DataFrame to DMatrix
    dtest = xgb.DMatrix(input_data)

    # Make a prediction
    try:
        prediction_prob = model.predict(dtest)[0]  # Get the probability prediction
        prediction_binary = 1 if prediction_prob > 0.5 else 0  # Convert probability to binary
        return jsonify({'CTR': prediction_binary})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
