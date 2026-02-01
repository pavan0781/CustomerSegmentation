import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load('customer_segmentation_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Customer Segmentation", page_icon="🛍")

st.title('🛍 Customer Segmentation App')
st.markdown("""
This application segments customers using a Machine Learning clustering model.
It helps businesses understand customer behavior and apply targeted marketing strategies.
""")

# Sidebar
with st.sidebar:
    st.header("Model Information")
    st.write("Algorithm: K-Means Clustering")
    st.write("Purpose: Customer behavior segmentation")

st.subheader("Enter Customer Details")

# User Inputs
col1, col2 = st.columns(2)

with col1:
    income = st.number_input('Annual Income (k$)', min_value=0, value=50)

with col2:
    spending = st.number_input('Spending Score (1–100)', min_value=1, max_value=100, value=50)

if st.button('Predict Segment', use_container_width=True):

    # Use dataset averages as base
    input_features = scaler.mean_.reshape(1, -1)

    # Map user inputs
    input_features[0, 3] = income
    input_features[0, 28] = spending

    # Scale and predict
    features_scaled = scaler.transform(input_features)
    cluster = model.predict(features_scaled)[0]

    st.divider()
    st.subheader(f"🎯 Result: Segment {cluster}")

    # Business Insights
    st.markdown("## 📊 Business Insights")

    if cluster == 0:
        st.success("""
        **High Value Customers**
        - High spending and strong purchase behavior
        - Ideal for loyalty programs and premium offers
        """)
    elif cluster == 1:
        st.warning("""
        **Price Sensitive Customers**
        - Lower spending behavior
        - Respond well to discounts and promotions
        """)
    elif cluster == 2:
        st.info("""
        **Occasional Customers**
        - Moderate activity
        - Need engagement campaigns
        """)
    else:
        st.error("""
        **Low Engagement Customers**
        - Rare purchases
        - Require reactivation strategies
        """)

    # Customer Position Chart
    st.markdown("## 📈 Customer Position")

    fig, ax = plt.subplots()
    ax.scatter(input_features[0, 3], input_features[0, 28], color='red', s=200)
    ax.set_xlabel("Income")
    ax.set_ylabel("Spending")
    ax.set_title("Customer Segmentation Map")
    st.pyplot(fig)

    # Segment Comparison Chart
    st.markdown("## 📊 Segment Comparison")

    segment_data = {
        "Segment 0": [80, 90],
        "Segment 1": [40, 30],
        "Segment 2": [60, 55],
        "Segment 3": [30, 20],
    }

    df_plot = pd.DataFrame(segment_data, index=["Income Level", "Spending Level"]).T
    st.bar_chart(df_plot)

    st.balloons()

st.markdown("---")
st.caption("Built using Machine Learning and Streamlit")
