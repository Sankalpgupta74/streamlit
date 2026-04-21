import streamlit as st
import pickle
import numpy as np
import os

st.title("ML Model Predictor")

# Load model safely
if not os.path.exists("model.pkl"):
    st.error("Model file not found!")
else:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    st.write("Enter feature values:")

    f1 = st.number_input("Feature 1")
    f2 = st.number_input("Feature 2")
    f3 = st.number_input("Feature 3")
    f4 = st.number_input("Feature 4")

    if st.button("Predict"):
        input_data = np.array([[f1, f2, f3, f4]])
        prediction = model.predict(input_data)
        st.success(f"Prediction: {prediction[0]}")
