import xgboost as xgb
import pandas as pd
import os

# Path to the model file
model_path = 'xgb_model.bin'

# Check if the model file exists
if not os.path.isfile(model_path):
    print(f"Model file '{model_path}' not found. Please check the file path.")
    exit()

# Load the model
try:
    model = xgb.Booster()
    model.load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load sample data (ensure this matches your input data structure)
sample_data = pd.read_csv('ad_10000records.csv')

# Process Timestamp if needed
if 'Timestamp' in sample_data.columns:
    sample_data['Timestamp'] = pd.to_datetime(sample_data['Timestamp'])
    sample_data['Year'] = sample_data['Timestamp'].dt.year
    sample_data['Month'] = sample_data['Timestamp'].dt.month
    sample_data['Day'] = sample_data['Timestamp'].dt.day
    sample_data['Hour'] = sample_data['Timestamp'].dt.hour

# Handle missing features in data
features = ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage',
            'Ad Topic Line', 'City', 'Gender', 'Country', 'Year', 'Month', 'Day', 'Hour']

for feature in features:
    if feature not in sample_data.columns:
        sample_data[feature] = 0  # Default value; adjust as needed

# Convert categorical features to numerical
categorical_features = ['Ad Topic Line', 'City', 'Gender', 'Country']
for feature in categorical_features:
    if feature in sample_data.columns:
        sample_data[feature] = sample_data[feature].astype('category').cat.codes

# Prepare data for prediction
X_test = sample_data[features]
dtest = xgb.DMatrix(X_test)

# Make a prediction
try:
    prediction = model.predict(dtest)
    print("Sample Prediction:", prediction)
except Exception as e:
    print(f"Error during prediction: {e}")
