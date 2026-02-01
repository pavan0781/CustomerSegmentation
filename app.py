import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('customer_segmentation_model.pkl') 
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Customer Segmentation", page_icon="🛍")

st.title('🛍 Customer Segmentation App')
st.write('Enter customer details to find their segment using the trained model.')

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("This app uses a K-Means model to group customers into segments based on behavior.")

# User inputs
col1, col2 = st.columns(2)

with col1:
    income = st.number_input('Annual Income (k$)', min_value=0, value=50)

with col2:
    spending = st.number_input('Spending Score (1–100)', min_value=1, max_value=100, value=50)

if st.button('Predict Segment', use_container_width=True):

    # Fill missing features using dataset averages
    input_features = scaler.mean_.copy().reshape(1, -1)

    # Map inputs to correct feature positions
    input_features[0, 3] = income
    input_features[0, 28] = spending

    # Scale
    features_scaled = scaler.transform(input_features)

    # Predict
    cluster = model.predict(features_scaled)[0]

    st.divider()
    st.subheader(f"Result: Segment {cluster}")

    if cluster == 0:
        st.write("**Strategy:** High value, frequent shoppers. Focus on loyalty rewards.")
    elif cluster == 1:
        st.write("**Strategy:** Budget-conscious shoppers. Focus on discounts and promotions.")
    else:
        st.write("**Strategy:** Occasional shoppers. Focus on re-engagement emails.")

    st.balloons()
