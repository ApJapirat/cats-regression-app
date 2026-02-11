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
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")

DATA_PATH = "data/car_prices_jordan.csv"
MODEL_PATH = "models/car_price_model.joblib"
META_PATH = "models/metadata.json"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the same features as in training."""
    import re

    def extract_year(model: str):
        m = re.search(r"(19|20)\d{2}", str(model))
        return int(m.group()) if m else np.nan

    def extract_cc(power: str):
        m = re.search(r"(\d+)\s*CC", str(power).upper())
        return float(m.group(1)) if m else np.nan

    out = df.copy()
    out["Price"] = out["Price"].astype(str).str.replace(",", "", regex=False).astype(float)

    out["Brand"] = out["Model"].astype(str).str.strip().str.split().str[0]
    out["Year"] = out["Model"].apply(extract_year)
    out["Property"] = out["Property"].astype(str).str.strip().str.lower()
    out["PowerCC"] = out["Power"].apply(extract_cc)
    out["Turbo"] = out["Power"].astype(str).str.contains("TURBO", case=False, na=False).astype(int)
    return out


def train_and_save():
    df = pd.read_csv(DATA_PATH)
    df = build_features(df)

    features = ["Brand", "Property", "Year", "PowerCC", "Turbo"]
    target = "Price"

    df = df.dropna(subset=features + [target]).copy()
    X = df[features]
    y = df[target].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    num_cols = ["Year", "PowerCC", "Turbo"]
    cat_cols = ["Brand", "Property"]

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

    meta = {
        "features": features,
        "target": target,
        "brand_options": sorted(df["Brand"].unique().tolist()),
        "property_options": sorted(df["Property"].unique().tolist()),
        "metrics": {"r2": float(r2), "mae": float(mae), "rmse": float(rmse)},
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
        return train_and_save()


# UI
st.title("🚗 Car Price Predictor (Jordan)")
st.caption("Multiple Regression (Linear) using 5 features: Brand, Property, Year, PowerCC, Turbo")

pipe, meta = load_or_train()

with st.sidebar:
    st.subheader("Model Info")
    st.write("Target:", meta["target"])
    st.write("Features:", ", ".join(meta["features"]))

    st.write("Test Metrics (hold-out test set):")
    st.metric("R²", f"{meta['metrics']['r2']:.3f}")
    st.metric("MAE", f"{meta['metrics']['mae']:.0f}")
    st.metric("RMSE", f"{meta['metrics']['rmse']:.0f}")

    if st.button("🔁 Retrain model (cloud)"):
        load_or_train.clear()
        pipe, meta = train_and_save()
        st.success("Retrained! (ลองกด Rerun / Refresh หน้าเว็บอีกที)")

st.subheader("Input Features")

# dropdown แบบกรอง 
df_raw = pd.read_csv(DATA_PATH).copy()
df_raw["Model"] = df_raw["Model"].astype(str).str.strip()
df_raw["Property"] = df_raw["Property"].astype(str).str.strip().str.lower()

# สร้าง Brand จากคำแรกของ Model
df_raw["Brand"] = df_raw["Model"].str.split().str[0]

# ดึง year จาก Model 
df_raw["Year_in_text"] = df_raw["Model"].str.extract(r"((?:19|20)\d{2})")[0]
df_raw["Year_in_text"] = pd.to_numeric(df_raw["Year_in_text"], errors="coerce")

# ดึง PowerCC + Turbo จาก Power 
df_raw["PowerCC_in_text"] = df_raw["Power"].astype(str).str.extract(r"(\d+)\s*CC", expand=False)
df_raw["PowerCC_in_text"] = pd.to_numeric(df_raw["PowerCC_in_text"], errors="coerce")
df_raw["Turbo_in_text"] = df_raw["Power"].astype(str).str.contains("TURBO", case=False, na=False).astype(int)

# Brand
brand_options = sorted(df_raw["Brand"].dropna().unique().tolist())
brand = st.selectbox("Brand", options=brand_options)

# Model
df_b = df_raw[df_raw["Brand"] == brand].copy()
model_options = sorted(df_b["Model"].dropna().unique().tolist())
model = st.selectbox("Model", options=model_options)

# Property 
df_m = df_b[df_b["Model"] == model].copy()
property_options = sorted(df_m["Property"].dropna().unique().tolist())
prop = st.selectbox("Property (transmission)", options=property_options)

# กรองเพิ่มตาม model+property เพื่อเอาค่า default ปี/cc/turbo
df_mp = df_m[df_m["Property"] == prop].copy()

# default ปี 
year_default = int(df_mp["Year_in_text"].dropna().iloc[0]) if df_mp["Year_in_text"].notna().any() else 2020
year = st.number_input("Year", min_value=1990, max_value=2026, value=year_default, step=1)

# default cc
cc_default = float(df_mp["PowerCC_in_text"].dropna().median()) if df_mp["PowerCC_in_text"].notna().any() else 1500.0
power_cc = st.number_input("Power (CC)", min_value=0.0, max_value=8000.0, value=float(cc_default), step=50.0)

# default turbo
turbo_default = int(df_mp["Turbo_in_text"].dropna().iloc[0]) if df_mp["Turbo_in_text"].notna().any() else 0
turbo = st.selectbox("Turbo", options=[0, 1], index=turbo_default,
                     format_func=lambda x: "Yes" if x == 1 else "No")

if st.button("Predict Price", type="primary"):
    input_df = pd.DataFrame([{
        "Brand": brand,
        "Property": prop,
        "Year": year,
        "PowerCC": power_cc,
        "Turbo": turbo,
    }])

    pred = float(pipe.predict(input_df)[0])
    st.success(f"✅ Predicted Price: **{pred:,.0f} JOD**")

    with st.expander("Show input data"):
        st.dataframe(input_df)

    with st.expander("Show selected raw rows (for sanity check)"):
        st.dataframe(df_mp[["Model", "Brand", "Property", "Power", "Price"]].head(20))
