import streamlit as st
import pickle
import numpy as np
import os

st.set_page_config(page_title="ML Model Predictor", layout="centered")

# Title
st.title("ML Model Predictor")

# Load model safely
if not os.path.exists("model.pkl"):
    st.error("❌ model.pkl file not found! Please upload it to GitHub.")
else:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    st.write("Enter feature values:")

    # 🔥 SAME features jo tumne model me use kiye
    temperature = st.number_input("Temperature", value=0.0)
    vibration = st.number_input("Vibration", value=0.0)
    humidity = st.number_input("Humidity", value=0.0)
    pressure = st.number_input("Pressure", value=0.0)

    if st.button("Predict"):
        try:
            input_data = np.array([[temperature, vibration, humidity, pressure]])
            prediction = model.predict(input_data)

            st.success(f"Prediction: {prediction[0]}")

        except Exception as e:
            st.error(f"Error: {e}")
