# app/streamlit_app.py
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

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

# features ที่หน้าเว็บใช้
EXPECTED_FEATURES = ["Age", "Breed", "Gender"]
TARGET_COL = "Weight"


def _load_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # rename columns ให้ชัวร์
    df = df.rename(columns={"Age (Years)": "Age", "Weight (kg)": "Weight"})

    # clean categorical
    for c in ["Breed", "Color", "Gender"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df


def train_and_save():
    """Train model in current runtime and save model+metadata."""
    df = _load_df()

    # เลือก feature / target
    features = EXPECTED_FEATURES
    target = TARGET_COL

    # กัน missing columns
    missing_cols = [c for c in (features + [target]) if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing columns: {missing_cols}")

    df = df.dropna(subset=features + [target]).copy()

    X = df[features]
    y = df[target].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # preprocess
    num_cols = ["Age"]
    cat_cols = ["Breed", "Gender"]

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
        "trained_on": "streamlit_cloud_runtime",
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return pipe, metadata


@st.cache_resource
def load_or_train():
    """
    Load saved model.
    If model/metadata missing, incompatible, or feature mismatch -> retrain.
    """
    try:
        pipe = joblib.load(MODEL_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # ถ้า metadata ไม่ตรงกับ features ที่โค้ดใช้ retrain
        if meta.get("features") != EXPECTED_FEATURES or meta.get("target") != TARGET_COL:
            raise ValueError("Feature/target mismatch -> retrain")

        return pipe, meta

    except Exception:
        pipe, meta = train_and_save()
        return pipe, meta


# UI
st.title("🐾 Cat Weight Predictor (Regression)")
st.caption("Predict cat weight (kg) using Age, Breed, Gender (Linear Regression)")

pipe, meta = load_or_train()

with st.sidebar:
    st.subheader("Model Info")
    st.write(f"Target: **{meta['target']}**")
    st.write("Features: " + ", ".join(meta["features"]))

    m = meta.get("metrics", {})
    st.markdown("**Test Metrics (hold-out test set):**")
    st.write(f"- R²  : {m.get('r2', 0):.3f}")
    st.write(f"- MAE : {m.get('mae', 0):.3f}")
    st.write(f"- RMSE: {m.get('rmse', 0):.3f}")
    st.caption("Note: This is a simple Linear Regression baseline.")

    if st.button("🔁 Retrain model (cloud)", help="If you update data or change features, retrain here."):
        load_or_train.clear()
        pipe, meta = train_and_save()
        st.success("Retrained successfully! Please refresh/rerun if needed.")

st.subheader("Input Features")

age = st.number_input("Age (Years)", min_value=0.0, max_value=30.0, value=3.0, step=0.5)
breed = st.selectbox("Breed", options=meta["breed_options"])
gender = st.selectbox("Gender", options=meta["gender_options"])

if st.button("Predict Weight (kg)", type="primary"):
    input_df = pd.DataFrame([{
        "Age": float(age),
        "Breed": str(breed),
        "Gender": str(gender),
    }])

    pred = float(pipe.predict(input_df)[0])
    st.success(f"✅ Predicted Weight: **{pred:.2f} kg**")

    with st.expander("Show input data"):
        st.dataframe(input_df, use_container_width=True)
