import streamlit as st
import pandas as pd
import xgboost as xgb
import os

# Path to the model file
model_path = 'xgb_model.bin'
if not os.path.isfile(model_path):
    st.error(f"Model file '{model_path}' not found. Please check the file path.")
    st.stop()

# Load the model
try:
    model = xgb.Booster()
    model.load_model(model_path)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit app
st.title("Click-Through Rate (CTR) Prediction")

# Input fields
daily_time_spent_on_site = st.number_input("Daily Time Spent on Site", min_value=0.0, format="%.2f")
age = st.number_input("Age", min_value=0)
area_income = st.number_input("Area Income", min_value=0.0, format="%.2f")
daily_internet_usage = st.number_input("Daily Internet Usage", min_value=0.0, format="%.2f")
ad_topic_line = st.text_input("Ad Topic Line")
city = st.text_input("City")
gender = st.selectbox("Gender", ["Male", "Female"])
country = st.text_input("Country")

# Timestamp handling
timestamp = st.date_input("Timestamp", value=pd.to_datetime("today"))

# Prepare the input data
if st.button("Predict"):
    try:
        # Create input DataFrame
        input_data = pd.DataFrame({
            'Daily Time Spent on Site': [daily_time_spent_on_site],
            'Age': [age],
            'Area Income': [area_income],
            'Daily Internet Usage': [daily_internet_usage],
            'Ad Topic Line': [ad_topic_line],
            'City': [city],
            'Gender': [gender],
            'Country': [country],
            'Timestamp': [timestamp]
        })

        # Process Timestamp if needed
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
        prediction_prob = model.predict(dtest)[0]  # Get the probability prediction
        prediction_binary = 1 if prediction_prob > 0.5 else 0  # Convert probability to binary

        # Display the result
        st.write(f"Predicted CTR: {'Yes' if prediction_binary == 1 else 'No'}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
