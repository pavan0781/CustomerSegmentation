import streamlit as st
import numpy as np
import joblib  # <--- Ensure this line is exactly like this

# Load model and scaler
model = joblib.load('customer_segmentation_model.pkl') 
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Customer Segmentation", page_icon="🛍")

st.title('🛍 Customer Segmentation App')
st.write('Enter customer details to find their segment using the trained model.')

# Sidebar for information
with st.sidebar:
    st.header("About")
    st.info("This app uses a K-Means model to group customers into segments based on behavior.") [cite: 9]

# User inputs
col1, col2 = st.columns(2)

with col1:
    income = st.number_input('Annual Income (k$)', min_value=0, value=50)

with col2:
    spending = st.number_input('Spending Score (1–100)', min_value=1, max_value=100, value=50)

if st.button('Predict Segment', use_container_width=True):
    # 1. Get the number of features the scaler expects (29) [cite: 7]
    num_features = scaler.n_features_in_ 

    # 2. Use the scaler's 'mean_' as the baseline baseline [cite: 4, 5]
    # This prevents 'zero-bias' where the model thinks all other features are 0.
    # It fills the 27 missing features with average values from your dataset.
    input_features = scaler.mean_.copy().reshape(1, -1) 

    # 3. Map user input to specific indices 
    # Based on the original feature list:
    # 'Income' is at index 3
    # 'Total_Spend' (Spending Score) is at index 28
    input_features[0, 3] = income
    input_features[0, 28] = spending

    # 4. Scale the features using the loaded scaler [cite: 7]
    features_scaled = scaler.transform(input_features)

    # 5. Predict the cluster using the loaded K-Means model [cite: 9]
    cluster = model.predict(features_scaled)[0]

    # Display Result
    st.divider()
    st.subheader(f"Result: Segment {cluster}")
    
    # Simple descriptions based on typical K-Means outputs for this dataset
    if cluster == 0:
        st.write("**Strategy:** High value, frequent shoppers. Focus on loyalty rewards.")
    elif cluster == 1:
        st.write("**Strategy:** Budget-conscious shoppers. Focus on discounts and promotions.")
    else:
        st.write("**Strategy:** Occasional shoppers. Focus on re-engagement emails.")

    st.balloons()
