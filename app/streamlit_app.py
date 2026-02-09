# app/streamlit_app.py
import os
import json
import joblib
import pandas as pd
import streamlit as st
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Cat Weight Predictor", page_icon="🐾")

# Paths
DATA_PATH = "data/cats_dataset.csv"
MODEL_PATH = "models/cat_weight_model.joblib"
META_PATH = "models/metadata.json"


def train_and_save():
    """Train Linear Regression and save model+metadata (runs on Streamlit Cloud too)."""
    df = pd.read_csv(DATA_PATH)

    # rename columns
    df = df.rename(columns={"Age (Years)": "Age", "Weight (kg)": "Weight"})

    # clean categorical
    for c in ["Breed", "Gender"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    #ลด noise: ตัด Color ออก
    features = ["Age", "Breed", "Gender"]
    target = "Weight"

    df = df.dropna(subset=features + [target]).copy()

    X = df[features]
    y = df[target].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    num_cols = ["Age"]
    cat_cols = ["Breed", "Gender"]

    #Linear Regression and Categorical
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", LinearRegression())
    ])

    pipe.fit(X_train, y_train)

    # evaluate
    y_pred = pipe.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)

    metadata = {
        "features": features,
        "target": target,
        "breed_options": sorted(df["Breed"].unique().tolist()),
        "gender_options": sorted(df["Gender"].unique().tolist()),
        "metrics": {"r2": r2, "mae": mae, "rmse": rmse},
        "trained_on": "streamlit_cloud_runtime"
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return pipe, metadata


@st.cache_resource
def load_or_train():
    """Load saved model. If incompatible/missing -> retrain in this environment."""
    try:
        pipe = joblib.load(MODEL_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return pipe, meta
    except Exception:
        pipe, meta = train_and_save()
        return pipe, meta


st.title("🐾 Cat Weight Predictor (Regression)")
st.caption("Predict cat weight (kg) using Age, Breed, Gender (Linear Regression)")

# load model/metadata (or retrain if broken)
pipe, meta = load_or_train()

with st.sidebar:
    st.subheader("Model Info")
    st.write("Target:", meta["target"])
    st.write("Features:", ", ".join(meta["features"]))

    st.write("Test Metrics (on hold-out test set):")
    m = meta["metrics"]
    st.write(f"- R²  : {m['r2']:.3f}")
    st.write(f"- MAE : {m['mae']:.3f}")
    st.write(f"- RMSE: {m['rmse']:.3f}")

    st.caption("Note: This is a simple Linear Regression baseline. Performance depends on dataset patterns.")

    if st.button("🔁 Retrain model (cloud)", help="If you update data, retrain the model."):
        load_or_train.clear()
        pipe, meta = train_and_save()
        st.success("Retrained successfully! Please rerun if needed.")

st.subheader("Input Features")

age = st.number_input("Age (Years)", min_value=0.0, max_value=30.0, value=3.0, step=0.5)
breed = st.selectbox("Breed", options=meta["breed_options"])
gender = st.selectbox("Gender", options=meta["gender_options"])

if st.button("Predict Weight (kg)", type="primary"):
    input_df = pd.DataFrame([{
        "Age": age,
        "Breed": breed,
        "Gender": gender
    }])

    pred = float(pipe.predict(input_df)[0])
    st.success(f"✅ Predicted Weight: **{pred:.2f} kg**")

    with st.expander("Show input data"):
        st.dataframe(input_df)
