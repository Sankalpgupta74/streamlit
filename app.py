import streamlit as st
import pickle
import numpy as np
import os

st.set_page_config(page_title="ML Predictor", layout="centered")
st.title("ML Model Predictor")

# Load model
if not os.path.exists("model.pkl"):
    st.error("Model file not found!")
else:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    st.subheader("Enter feature values:")

    # 🔥 Dynamic feature detection
    try:
        feature_names = model.feature_names_in_
    except:
        feature_names = [f"Feature {i+1}" for i in range(model.n_features_in_)]

    inputs = []

    for feature in feature_names:
        val = st.number_input(feature, value=0.0)
        inputs.append(val)

    if st.button("Predict"):
        input_data = np.array([inputs])
        prediction = model.predict(input_data)
        st.success(f"Prediction: {prediction[0]}")
