import streamlit as st
import numpy as np
import joblib

# Load model and scaler
# Note: Ensure these .pkl files are uploaded to your GitHub repo!
model = joblib.load('customer_segmentation_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('🛍 Customer Segmentation App')

st.write('Enter customer details to find their segment.')

# User inputs
income = st.number_input('Annual Income (k$)', min_value=0)
spending = st.number_input('Spending Score (1–100)', min_value=0, max_value=100)

if st.button('Predict Segment'):
    # 1. Get the number of features the scaler expects (29)
    num_features = scaler.n_features_in_

    # 2. Use the scaler's 'mean_' as the baseline instead of zeros
    # This represents a "typical" customer for all fields the user didn't fill in.
    input_features = scaler.mean_.copy().reshape(1, -1)

    # 3. Overwrite the specific indices for Income and Total_Spend
    # Indices based on your mapping: Income (3), Total_Spend (28)
    input_features[0, 3] = income
    input_features[0, 28] = spending

    # 4. Scale the features
    # Standard scaler applies: (x - mean) / stdev
    features_scaled = scaler.transform(input_features)

    # 5. Predict
    cluster = model.predict(features_scaled)[0]

    # Display Result
    st.balloons()
    st.success(f"Targeting Strategy: Customer belongs to **Segment {cluster}**")
