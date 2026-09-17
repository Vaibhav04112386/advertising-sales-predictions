import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("linear_regression_model.joblib")

# Page title
st.title("Advertising Sales Prediction")

st.write("Enter the advertising expenditure to predict sales.")

# Input fields
tv = st.number_input(
    "TV Advertising Expenditure",
    min_value=0.0,
    value=0.0
)

radio = st.number_input(
    "Radio Advertising Expenditure",
    min_value=0.0,
    value=0.0
)

newspaper = st.number_input(
    "Newspaper Advertising Expenditure",
    min_value=0.0,
    value=0.0
)

# Prediction button
if st.button("Predict Sales"):

    # Create input DataFrame
    input_data = pd.DataFrame(
        [[tv, radio, newspaper]],
        columns=["TV", "Radio", "Newspaper"]
    )

    # Make prediction
    prediction = model.predict(input_data)

    # Display result
    st.success(
        f"Predicted Sales: {prediction[0]:.2f}"
    )
