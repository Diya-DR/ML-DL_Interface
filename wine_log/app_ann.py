
import streamlit as st
import numpy as np
import joblib


# --------------------------------
# 1. Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Wine Classification",
    page_icon="🍷",
    layout="wide"
)


# --------------------------------
# 2. Load ANN Model
# --------------------------------

model = joblib.load("wine_model_ann.pkl")


# --------------------------------
# 3. Application Title
# --------------------------------

st.title("🍷 Wine Classification System")

st.write(
    "Predict wine type based on its 13 chemical properties "
    "using an Artificial Neural Network."
)

st.divider()


# --------------------------------
# 4. Input Section
# --------------------------------

st.subheader("Enter Wine Details")

col1, col2, col3 = st.columns(3)


with col1:
    alcohol = st.number_input(
        "Alcohol", value=13.0, step=0.1
    )

    malic_acid = st.number_input(
        "Malic Acid", value=2.0, step=0.1
    )

    ash = st.number_input(
        "Ash", value=2.3, step=0.1
    )

    alcalinity = st.number_input(
        "Alcalinity of Ash", value=19.0, step=0.1
    )

    magnesium = st.number_input(
        "Magnesium", value=100.0, step=1.0
    )


with col2:
    total_phenols = st.number_input(
        "Total Phenols", value=2.0, step=0.1
    )

    flavanoids = st.number_input(
        "Flavanoids", value=2.0, step=0.1
    )

    nonflavanoid = st.number_input(
        "Nonflavanoid Phenols", value=0.3, step=0.01
    )

    proanthocyanins = st.number_input(
        "Proanthocyanins", value=1.5, step=0.1
    )


with col3:
    color_intensity = st.number_input(
        "Color Intensity", value=5.0, step=0.1
    )

    hue = st.number_input(
        "Hue", value=1.0, step=0.01
    )

    od280 = st.number_input(
        "OD280/OD315 of Diluted Wines", value=3.0, step=0.1
    )

    proline = st.number_input(
        "Proline", value=1000.0, step=1.0
    )


st.divider()


# --------------------------------
# 5. Prediction
# --------------------------------

if st.button(
    "Predict Wine Type",
    type="primary",
    use_container_width=True
):

    # Prepare all 13 input features
    input_data = np.array([[
        alcohol,
        malic_acid,
        ash,
        alcalinity,
        magnesium,
        total_phenols,
        flavanoids,
        nonflavanoid,
        proanthocyanins,
        color_intensity,
        hue,
        od280,
        proline
    ]])

    try:

        # --------------------------------
        # First Hidden Layer
        # --------------------------------

        z1 = np.dot(input_data, model['w1']) + model['b1']

        # ReLU Activation
        a1 = np.maximum(0, z1)


        # --------------------------------
        # Output Layer
        # --------------------------------

        z2 = np.dot(a1, model['w2']) + model['b2']


        # --------------------------------
        # Prediction
        # --------------------------------

        prediction = np.argmax(z2, axis=1)[0]


        # --------------------------------
        # Display Result
        # --------------------------------

        st.success("Prediction Completed!")

        st.subheader("Prediction Result")

        st.metric(
            label="Predicted Wine Class",
            value=f"Class {prediction}"
        )


        # Display output scores
        st.subheader("Class Scores")

        for i, score in enumerate(z2[0]):
            st.write(f"Class {i}: {score:.4f}")


    except Exception as e:

        st.error(f"Prediction Error: {e}")


# --------------------------------
# 6. Footer
# --------------------------------

st.divider()

st.caption(
    "Wine Classification System | "
    "Artificial Neural Network"
)
