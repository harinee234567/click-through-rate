import xgboost as xgb
import pandas as pd
import os

# Path to the dataset
data_path = 'ad_10000records.csv'

# Load the dataset
if not os.path.isfile(data_path):
    print(f"Data file '{data_path}' not found. Please check the file path.")
    exit()

data = pd.read_csv(data_path)

# Print columns to verify
print("Columns in data:", data.columns)

# Define features and target
features = [
    'Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage',
    'Ad Topic Line', 'City', 'Gender', 'Country', 'Year', 'Month', 'Day', 'Hour'
]
target = 'Clicked on Ad'

# Process Timestamp if needed
if 'Timestamp' in data.columns:
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data['Year'] = data['Timestamp'].dt.year
    data['Month'] = data['Timestamp'].dt.month
    data['Day'] = data['Timestamp'].dt.day
    data['Hour'] = data['Timestamp'].dt.hour

# Handle missing features in data
for feature in features:
    if feature not in data.columns:
        data[feature] = 0  # Default value; adjust as needed

# Prepare data for training
X = data[features]
y = data[target]

# Convert categorical features to numerical (if needed)
categorical_features = ['Ad Topic Line', 'City', 'Gender', 'Country']
for feature in categorical_features:
    if feature in X.columns:
        X[feature] = X[feature].astype('category').cat.codes

# Initialize and train the model
model = xgb.XGBClassifier()

try:
    model.fit(X, y)
    model.save_model('xgb_model.bin')
    print("Model trained and saved successfully.")
except Exception as e:
    print(f"Error during model training: {e}")
