import streamlit as st
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load("iris_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Iris Flower Classification", layout="centered")

st.title("🌸 Iris Flower Classification")
st.write("Predict the species of an Iris flower using Machine Learning")

# User inputs
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, format="%.2f")
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, format="%.2f")
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, format="%.2f")
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, format="%.2f")

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    species = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    st.success(f"🌼 Predicted Species: **{species[prediction[0]]}**")


st.markdown("---")
st.markdown("**Author:** Zoha Fatima")
