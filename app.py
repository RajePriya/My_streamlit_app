import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

# Load the model and encoders
# Load the model and encoders
xgb_model = joblib.load('xgboost_model.pkl')
scaler = joblib.load('standard_scaler.pkl')
encoder = joblib.load('onehot_encoder.pkl')


# Title of the app
st.title('Crop Production Prediction')

# Input form for user inputs
with st.form("input_form"):
    temperature = st.number_input('Temperature', min_value=-50.0, max_value=50.0, value=25.0)
    rainfall = st.number_input('Rainfall', min_value=0.0, max_value=1000.0, value=100.0)
    humidity = st.number_input('Humidity', min_value=0.0, max_value=100.0, value=50.0)
    sun_hours = st.number_input('Sun hours', min_value=0.0, max_value=24.0, value=8.0)
    season = st.selectbox('Season', options=encoder.categories_[0])
    crop = st.selectbox('Crop', options=encoder.categories_[1])
    submit = st.form_submit_button("Predict")

if submit:
    # Feature Engineering
    temp_rainfall_ratio = temperature / rainfall if rainfall != 0 else 0
    humidity_sunratio = humidity / sun_hours if sun_hours != 0 else 0

    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'Temperature': [temperature],
        'Rainfall': [rainfall],
        'Humidity': [humidity],
        'Sun hours': [sun_hours],
        'Temp_Rainfall_Ratio': [temp_rainfall_ratio],
        'Humidity_SunRatio': [humidity_sunratio],
        'Season': [season],
        'Crop': [crop]
    })

    # Encode categorical features
    encoded_features = encoder.transform(input_data[['Season', 'Crop']])
    encoded_columns = list(encoder.get_feature_names_out(['Season', 'Crop']))
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_columns)

    # Combine the input data with encoded features
    input_data = pd.concat([input_data.drop(columns=['Season', 'Crop']), encoded_df], axis=1)

    # Scale the input features
    scaled_input = scaler.transform(input_data)

    # Predict the production
    prediction = xgb_model.predict(scaled_input)

    # Display the prediction
    st.write(f'Predicted Production: {prediction[0]:.2f}')
    