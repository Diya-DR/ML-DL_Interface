
import streamlit as st
import pandas as pd
import pickle


# --------------------------------
# 1. Load the trained model
# --------------------------------

MODEL_PATH = "wine_logistic_model.pkl"

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


# --------------------------------
# 2. Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="centered"
)

st.title("🍷 Wine Quality Prediction")
st.write(
    "Predict the quality of wine using a "
    "Machine Learning Logistic Regression model."
)

st.divider()


# --------------------------------
# 3. Input Features
# --------------------------------

st.subheader("Enter Wine Characteristics")

col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input(
        "Fixed Acidity",
        min_value=0.0,
        value=7.0,
        step=0.1
    )

    volatile_acidity = st.number_input(
        "Volatile Acidity",
        min_value=0.0,
        value=0.5,
        step=0.01
    )

    citric_acid = st.number_input(
        "Citric Acid",
        min_value=0.0,
        value=0.3,
        step=0.01
    )



    chlorides = st.number_input(
        "Chlorides",
        min_value=0.0,
        value=0.08,
        step=0.001,
        format="%.3f"
    )

    free_sulfur_dioxide = st.number_input(
        "Free Sulfur Dioxide",
        min_value=0.0,
        value=15.0,
        step=1.0
    )

with col2:

    density = st.number_input(
        "Density",
        min_value=0.0,
        value=0.996,
        step=0.001,
        format="%.4f"
    )


    sulphates = st.number_input(
        "Sulphates",
        min_value=0.0,
        value=0.65,
        step=0.01
    )

    alcohol = st.number_input(
        "Alcohol",
        min_value=0.0,
        value=10.0,
        step=0.1
    )


# --------------------------------
# 4. Prediction
# --------------------------------

st.divider()

if st.button("Predict Wine Quality", type="primary",
             use_container_width=True):

    # Create DataFrame with the same feature names
    input_data = pd.DataFrame([{
        "fixed acidity": fixed_acidity,
        "volatile acidity": volatile_acidity,
        "citric acid": citric_acid,
        "chlorides": chlorides,
        "total sulfur dioxide": free_sulfur_dioxide,
        "density": density,
        "sulphates": sulphates,
        "alcohol": alcohol
    }])

    try:
        # Make prediction
        result = ""
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            result = "Good"
        else:
            result = "Bad"

        st.subheader("Prediction Result")

        st.success(f"Predicted Wine Quality: {result}")

        # Display probability if supported
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]

            st.write("Prediction Confidence")

            probability_df = pd.DataFrame({
                "Wine Quality": model.classes_,
                "Probability": probabilities
            })

            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )

    except Exception as e:
        st.error(f"Prediction Error: {e}")

st.divider()

st.caption(
    "Powered by Machine Learning | Logistic Regression"
)
