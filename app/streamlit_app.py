import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Cat Weight Predictor", page_icon="🐾")

st.title("🐾 Cat Weight Predictor (Regression)")
st.caption("Predict cat weight (kg) using Age, Breed, Color, Gender")

# โหลดโมเดล + metadata
MODEL_PATH = "models/cat_weight_model.joblib"
META_PATH = "models/metadata.json"

pipe = joblib.load(MODEL_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

breed_options = meta["breed_options"]
color_options = meta["color_options"]
gender_options = meta["gender_options"]

with st.sidebar:
    st.subheader("Model Info")
    st.write("Target:", meta["target"])
    st.write("Features:", ", ".join(meta["features"]))
    st.write("Test Metrics:")
    st.write(meta["metrics"])

st.subheader("Input Features")

age = st.number_input("Age (Years)", min_value=0.0, max_value=30.0, value=3.0, step=0.5)
breed = st.selectbox("Breed", options=breed_options)
color = st.selectbox("Color", options=color_options)
gender = st.selectbox("Gender", options=gender_options)

if st.button("Predict Weight (kg)", type="primary"):
    input_df = pd.DataFrame([{
        "Age": age,
        "Breed": breed,
        "Color": color,
        "Gender": gender
    }])

    pred = pipe.predict(input_df)[0]
    st.success(f"✅ Predicted Weight: **{pred:.2f} kg**")

    with st.expander("Show input data"):
        st.dataframe(input_df)
