
import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('customer_segmentation_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('🛍 Customer Segmentation App')

st.write('Enter customer details to find their segment.')

income = st.number_input('Annual Income (k$)', min_value=0)
spending = st.number_input('Spending Score (1–100)', min_value=0, max_value=100)

if st.button('Predict Segment'):
    # Ensure the input features match the number of features the scaler was trained on
    # The original scaler was trained on 29 features. Need to reconstruct that input.
    # For this example, we'll assume income and spending are the only relevant features for a simplified model.
    # In a real scenario, you would need to collect all 29 features from the user or calculate them.
    # For demonstration, let's create dummy features for the missing ones.
    # This is a placeholder and should be replaced with actual feature engineering logic if deployed.

    # Get the number of features the scaler expects
    num_features_expected_by_scaler = scaler.mean_.shape[0] # or scaler.n_features_in_

    # Create an array of zeros with the expected number of features
    input_features = np.zeros((1, num_features_expected_by_scaler))

    # Place 'income' and 'spending' in their respective (hypothetical) positions
    # This part needs to be adjusted based on the *actual* order of features when scaler was fit.
    # For simplicity, assuming income is the 3rd feature (index 2) and spending is a new feature.
    # For the provided model and scaler, income was the 3rd column, and 'Total_Spend' was the last.
    # Let's assume 'spending' maps to 'Total_Spend' for this simplified example.

    # From the original notebook, feature_names was:
    # ['Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome', 'Teenhome', 'Recency',
    # 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
    # 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
    # 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain',
    # 'Z_CostContact', 'Z_Revenue', 'Response', 'Customer_Tenure', 'Total_Spend']

    # Let's map user input to the correct features based on `feature_names`.
    # Assuming 'Income' is at index 3 and 'Total_Spend' is at index 28 (last one)
    feature_map = {
        'Income': 3,
        'Total_Spend': 28 # This is `spending` from the user input
    }

    # Fill in the known values. Other features will remain zero or a default/median value.
    # This is a simplification; a production app would require a more robust feature input process.
    input_features[0, feature_map['Income']] = income
    input_features[0, feature_map['Total_Spend']] = spending

    # Scale the features
    features_scaled = scaler.transform(input_features)

    # Predict the cluster
    cluster = model.predict(features_scaled)[0]

    st.success(f"Customer belongs to Segment {cluster}")
