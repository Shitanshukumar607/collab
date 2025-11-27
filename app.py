import streamlit as st
import pandas as pd
import pickle
import numpy as np


# Load the model and encoders
@st.cache_resource
def load_resources():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders


try:
    model, encoders = load_resources()
except FileNotFoundError:
    st.error(
        "Model or Encoder files not found. Please ensure 'best_model.pkl' and 'encoders.pkl' are in the same directory."
    )
    st.stop()

st.title("ASD Prediction App")
st.write("Enter the details below to predict the likelihood of ASD.")

# Input form
with st.form("prediction_form"):
    st.header("Screening Scores")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        a1 = st.selectbox("A1 Score", [0, 1])
        a6 = st.selectbox("A6 Score", [0, 1])
    with col2:
        a2 = st.selectbox("A2 Score", [0, 1])
        a7 = st.selectbox("A7 Score", [0, 1])
    with col3:
        a3 = st.selectbox("A3 Score", [0, 1])
        a8 = st.selectbox("A8 Score", [0, 1])
    with col4:
        a4 = st.selectbox("A4 Score", [0, 1])
        a9 = st.selectbox("A9 Score", [0, 1])
    with col5:
        a5 = st.selectbox("A5 Score", [0, 1])
        a10 = st.selectbox("A10 Score", [0, 1])

    st.header("Personal Details")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", min_value=0, max_value=100, value=18)
        gender = st.selectbox("Gender", encoders["gender"].classes_)
        ethnicity = st.selectbox("Ethnicity", encoders["ethnicity"].classes_)
        jaundice = st.selectbox("Born with Jaundice?", encoders["jaundice"].classes_)

    with c2:
        austim = st.selectbox("Family member with PDD?", encoders["austim"].classes_)
        contry_of_res = st.selectbox(
            "Country of Residence", encoders["contry_of_res"].classes_
        )
        used_app_before = st.selectbox(
            "Used App Before?", encoders["used_app_before"].classes_
        )
        relation = st.selectbox("Relation", encoders["relation"].classes_)
        result = st.number_input(
            "Result (Screening Score Total)", min_value=0.0, value=0.0
        )

    submit_button = st.form_submit_button("Predict")

if submit_button:
    # Prepare input data
    input_data = {
        "A1_Score": [a1],
        "A2_Score": [a2],
        "A3_Score": [a3],
        "A4_Score": [a4],
        "A5_Score": [a5],
        "A6_Score": [a6],
        "A7_Score": [a7],
        "A8_Score": [a8],
        "A9_Score": [a9],
        "A10_Score": [a10],
        "age": [int(age)],  # Model expects int for age
        "gender": [gender],
        "ethnicity": [ethnicity],
        "jaundice": [jaundice],
        "austim": [austim],
        "contry_of_res": [contry_of_res],
        "used_app_before": [used_app_before],
        "result": [result],
        "relation": [relation],
    }

    input_df = pd.DataFrame(input_data)

    # Preprocessing
    # Note: The cleaning steps in model.py (mapping countries, handling missing values)
    # are implicitly handled because we are selecting from the encoder classes which
    # represent the cleaned data. We just need to encode.

    # Apply Label Encoding
    for column, encoder in encoders.items():
        if column in input_df.columns:
            input_df[column] = encoder.transform(input_df[column])

    # Ensure column order matches training
    # We need to know the exact column order the model expects.
    # Based on model.py, X = df.drop(columns=["Class/ASD"])
    # The columns in df (after dropping ID and age_desc) are:
    # A1..A10, age, gender, ethnicity, jaundice, austim, contry_of_res, used_app_before, result, relation

    # Let's reorder to be safe.
    expected_cols = [
        "A1_Score",
        "A2_Score",
        "A3_Score",
        "A4_Score",
        "A5_Score",
        "A6_Score",
        "A7_Score",
        "A8_Score",
        "A9_Score",
        "A10_Score",
        "age",
        "gender",
        "ethnicity",
        "jaundice",
        "austim",
        "contry_of_res",
        "used_app_before",
        "result",
        "relation",
    ]

    input_df = input_df[expected_cols]

    # Predict
    prediction = model.predict(input_df)

    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error("The model predicts a high likelihood of ASD.")
    else:
        st.success("The model predicts a low likelihood of ASD.")
