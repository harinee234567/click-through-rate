import requests
import json
import numpy as np

def convert_numpy_types(data):
    """Convert numpy data types to native Python data types."""
    if isinstance(data, dict):
        return {key: convert_numpy_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(element) for element in data]
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data

url = 'http://127.0.0.1:5000/predict'
data = {
    "Daily Time Spent on Site": 49.21,
    "Age": 30,
    "Area Income": 54324.73,
    "Daily Internet Usage": 201.58,
    "Ad Topic Line": 484,
    "City": 495,
    "Gender": 0,
    "Country": 149,
    "Timestamp": "2016-07-21 10:54:35"  # Ensure this is correctly formatted
}

# Convert any numpy types to native Python types
data = convert_numpy_types(data)

try:
    # Send POST request
    response = requests.post(url, json=data)
    # Check for HTTP errors
    response.raise_for_status()

    # Print the raw response text for debugging
    print("Response Text:", response.text)

    # Attempt to parse the JSON response
    try:
        response_json = response.json()
        print("Response JSON:", response_json)
    except json.JSONDecodeError:
        print("Response content is not valid JSON.")

except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
