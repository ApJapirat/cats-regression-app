# app/streamlit_app.py
import os
import json
import joblib
import pandas as pd
import streamlit as st

import sys
sys.path.append(os.path.abspath("."))

from src.features import build_features
from app.ui import render_sidebar, render_inputs, render_output


st.set_page_config(page_title="Car Price Predictor (Jordan)", page_icon="🚗")

DATA_PATH = "data/car_prices_jordan.csv"
MODEL_PATH = "models/car_price_model.joblib"
META_PATH = "models/metadata.json"


def train_and_save_in_cloud():
    """Fallback retrain on Streamlit Cloud runtime if loading model fails."""
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    import numpy as np

    df_raw = pd.read_csv(DATA_PATH)
    df = build_features(df_raw)

    features = ["Brand", "Model", "Property", "Year", "PowerCC", "Turbo"]
    target = "Price"

    df = df.dropna(subset=features + [target]).copy()
    X = df[features]
    y = df[target].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    num_cols = ["Year", "PowerCC", "Turbo"]
    cat_cols = ["Brand", "Model", "Property"]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", Ridge(alpha=1.0)),
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)

    # dependent dropdown mappings
    brands = sorted(df["Brand"].dropna().unique().tolist())
    brand_to_models = {}
    brand_model_to_props = {}

    for b in brands:
        models = sorted(df.loc[df["Brand"] == b, "Model"].dropna().unique().tolist())
        brand_to_models[b] = models
        for m in models:
            props = sorted(df.loc[(df["Brand"] == b) & (df["Model"] == m), "Property"].dropna().unique().tolist())
            brand_model_to_props[f"{b}|||{m}"] = props

    meta = {
        "features": features,
        "target": target,
        "metrics": {"r2": float(r2), "mae": float(mae), "rmse": float(rmse)},
        "options": {
            "brands": brands,
            "brand_to_models": brand_to_models,
            "brand_model_to_props": brand_model_to_props,
        }
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return pipe, meta


@st.cache_resource
def load_or_train():
    try:
        pipe = joblib.load(MODEL_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return pipe, meta
    except Exception:
        return train_and_save_in_cloud()


# ===== Page =====
st.title("🚗 Car Price Predictor (Jordan)")
st.caption("Multiple Regression (Linear) with 6 features: Brand, Model, Property, Year, PowerCC, Turbo")

pipe, meta = load_or_train()

# sidebar
retrain_clicked = render_sidebar(meta)
if retrain_clicked:
    load_or_train.clear()
    pipe, meta = train_and_save_in_cloud()
    st.success("Retrained! (ลองกด Rerun / Refresh หน้าเว็บอีกที)")

# data for linked inputs + sanity table
df_raw = pd.read_csv(DATA_PATH)
df_feat = build_features(df_raw).dropna(subset=["Brand", "Model", "Property", "Year", "PowerCC", "Turbo", "Price"]).copy()

# inputs
input_df, subset = render_inputs(meta, df_feat)

# predict
if st.button("Predict Price", type="primary"):
    pred = float(pipe.predict(input_df)[0])
    render_output(pred, input_df, subset)
