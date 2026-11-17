
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# --------------------------------
# 1. Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide"
)

# --------------------------------
# 2. Load Saved Models
# --------------------------------

loaded_scaler = joblib.load("scaler.pkl")
loaded_kmeans = joblib.load("kmeans_model.pkl")
loaded_pca = joblib.load("pca_model.pkl")

# --------------------------------
# 3. Application Title
# --------------------------------

st.title("👥 Customer Segmentation System")

st.write(
    "Identify customer groups using Machine Learning "
    "and K-Means Clustering."
)

st.divider()

# --------------------------------
# 4. Customer Input
# --------------------------------

st.subheader("Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=25
    )

with col2:
    income = st.number_input(
        "Annual Income (k$)",
        min_value=1,
        max_value=200,
        value=50
    )

with col3:
    spending_score = st.slider(
        "Spending Score",
        min_value=1,
        max_value=100,
        value=50
    )

st.divider()

# --------------------------------
# 5. Prediction
# --------------------------------

if st.button("Predict Customer Segment", type="primary"):

    # Prepare input data
    input_data = pd.DataFrame(
        [[age, income, spending_score]],
        columns=["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    )

    # Scale input
    scaled_data = loaded_scaler.transform(input_data)

    # Predict cluster
    cluster = loaded_kmeans.predict(scaled_data)[0]

    # Apply PCA
    pca_data = loaded_pca.transform(scaled_data)

    # --------------------------------
    # 6. Display Results
    # --------------------------------

    st.success("Customer segment predicted successfully!")

    st.subheader("Prediction Results")

    col1, col2 = st.columns(2) 

    with col1:
        st.metric(
            "Assigned Cluster",
            f"Cluster {cluster}"
        )

    with col2:
        st.metric(
            "PCA Component 1",
            f"{pca_data[0][0]:.2f}"
        )

    # --------------------------------
    # 7. Customer Profile
    # --------------------------------

    st.subheader("Customer Profile")

    st.dataframe(input_data, use_container_width=True)


