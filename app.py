import streamlit as st
import pandas as pd
import xgboost as xgb
import os
from datetime import datetime

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

# Date input
timestamp_date = st.date_input("Date", value=pd.to_datetime("today").date())
# Time input
timestamp_time = st.time_input("Time", value=datetime(2023, 1, 1, 0, 0).time())  # Default to 00:00

# Combine date and time into a single timestamp
timestamp = datetime.combine(timestamp_date, timestamp_time)

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

        # Process Timestamp
        input_data['Year'] = input_data['Timestamp'].dt.year
        input_data['Month'] = input_data['Timestamp'].dt.month
        input_data['Day'] = input_data['Timestamp'].dt.day
        input_data['Hour'] = input_data['Timestamp'].dt.hour

        # Drop the original Timestamp column as it's no longer needed
        input_data.drop(columns=['Timestamp'], inplace=True)

        # Example mappings; replace with actual mappings or loading of encoders used during training
        ad_topic_line_mapping = {'Total 5thgeneration standardization': 484}
        city_mapping = {'West Richard': 495}
        gender_mapping = {'Male': 0, 'Female': 1}
        country_mapping = {'Qatar': 149}
        
        # Encode categorical features
        input_data['Ad Topic Line'] = input_data['Ad Topic Line'].map(ad_topic_line_mapping).fillna(-1).astype(int)
        input_data['City'] = input_data['City'].map(city_mapping).fillna(-1).astype(int)
        input_data['Gender'] = input_data['Gender'].map(gender_mapping).fillna(-1).astype(int)
        input_data['Country'] = input_data['Country'].map(country_mapping).fillna(-1).astype(int)

        # Ensure the order of columns matches the model's expected feature order
        features = ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage',
                    'Ad Topic Line', 'City', 'Gender', 'Country', 'Year', 'Month', 'Day', 'Hour']
        input_data = input_data[features]

        # Debugging: print input data
        st.write("Encoded Input Data:", input_data)

        # Convert DataFrame to DMatrix
        dtest = xgb.DMatrix(input_data)

        # Make a prediction
        prediction_prob = model.predict(dtest)[0]  # Get the probability prediction

        # Debugging: print prediction probability
        st.write("Prediction Probability:", prediction_prob)

        # Convert probability to binary
        prediction_binary = 1 if prediction_prob > 0.7 else 0

        # Display the result
        st.write(f"Predicted CTR: {'Yes, You have clicked on the AD' if prediction_binary == 1 else 'No, You have not clicked on the AD'}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
