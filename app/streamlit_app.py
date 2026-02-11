# app/streamlit_app.py
import os
import json
import joblib
import pandas as pd
import streamlit as st

import sys
sys.path.append(os.path.abspath("."))

from src.features import build_features
from app.ui import render_sidebar, render_inputs
from app.style import apply_dark_style


st.set_page_config(page_title="Car Price Predictor (Jordan)", page_icon="🚗", layout="wide")

DATA_PATH = "data/car_prices_jordan.csv"
MODEL_PATH = "models/car_price_model.joblib"
META_PATH = "models/metadata.json"


def train_and_save_in_cloud():
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


# ===== Style =====
apply_dark_style()

# ===== Load model/meta =====
pipe, meta = load_or_train()

# ===== Sidebar =====
retrain_clicked = render_sidebar(meta)
if retrain_clicked:
    load_or_train.clear()
    pipe, meta = train_and_save_in_cloud()
    st.success("Retrained! (ลองกด Rerun / Refresh หน้าเว็บอีกที)")

# ===== Header =====
st.markdown(
    f"""
    <div class="card glow">
      <div style="display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap;">
        <div>
          <div style="font-size: 2.0rem; font-weight: 800;">🚗 Car Price Predictor <span class="muted">(Jordan)</span></div>
          <div class="muted">Predict used-car price (JOD) using Brand, Model, Property, Year, PowerCC, Turbo</div>
        </div>
        <div style="text-align:right;">
          <div class="tiny">Model</div>
          <div style="font-weight:700;">Linear Regression (Ridge)</div>
          <div class="tiny">R²: {meta['metrics']['r2']:.3f} • RMSE: {meta['metrics']['rmse']:.0f}</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")  # spacing

# ===== Prepare data =====
df_raw = pd.read_csv(DATA_PATH)
df_feat = build_features(df_raw).dropna(subset=["Brand", "Model", "Property", "Year", "PowerCC", "Turbo", "Price"]).copy()

# ===== Main layout =====
col_left, col_right = st.columns([1.15, 1.0], gap="large")

with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    input_df, subset = render_inputs(meta, df_feat)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    predict_clicked = st.button("⚡ Predict Price", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="card glow">', unsafe_allow_html=True)
    st.subheader("Result")
    st.caption("Currency: Jordanian Dinar (JOD)")
    if "last_pred" not in st.session_state:
        st.session_state["last_pred"] = None

    if predict_clicked:
        pred = float(pipe.predict(input_df)[0])
        st.session_state["last_pred"] = pred

    pred = st.session_state["last_pred"]
    if pred is None:
        st.markdown("<div class='muted'>กรอกข้อมูลแล้วกด Predict เพื่อดูผลลัพธ์</div>", unsafe_allow_html=True)
    else:
        rmse = float(meta["metrics"]["rmse"])
        low, high = max(0.0, pred - rmse), pred + rmse
        st.markdown(
            f"""
            <div style="font-size:2.1rem; font-weight:900;">{pred:,.0f} JOD</div>
            <div class="muted">Estimated range (±RMSE): {low:,.0f} – {high:,.0f} JOD</div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ===== Tabs =====
tab1, tab2, tab3 = st.tabs(["📄 Dataset Preview", "📊 Insights", "ℹ️ About Model"])

with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("Selected rows (sanity check):")
    if len(subset):
        st.dataframe(subset[["Model", "Brand", "Property", "Power", "Year", "PowerCC", "Turbo", "Price"]].head(20),
                     use_container_width=True)
        st.caption(f"Rows matched: {len(subset)}")
    else:
        st.info("No exact row match for this selection (still ok — model can generalize).")
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    # กราฟเบา ๆ ไม่ต้องพึ่ง seaborn
    st.write("Quick insights (filtered by Brand if possible)")
    # filter to selected brand if exists
    brand_val = input_df.loc[0, "Brand"]
    df_plot = df_feat[df_feat["Brand"] == brand_val].copy()
    if len(df_plot) < 30:
        df_plot = df_feat.copy()

    st.write("Price distribution")
    st.bar_chart(df_plot["Price"].value_counts(bins=30).sort_index())

    st.write("Year vs Price (sample)")
    sample = df_plot[["Year", "Price"]].dropna().sample(min(800, len(df_plot)), random_state=42)
    st.scatter_chart(sample, x="Year", y="Price")
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        """
        **Pipeline**
        - Feature engineering: Brand/Year from `Model`, PowerCC/Turbo from `Power`
        - Preprocess: StandardScaler (numeric) + OneHotEncoder (categorical)
        - Model: Ridge Regression (linear regression family)

        **Metrics meaning**
        - **R²**: ใกล้ 1 ดี (อธิบายความแปรปรวนของราคาได้มากขึ้น)
        - **MAE/RMSE**: ใกล้ 0 ดี (error น้อยลง)

        **Limitations**
        - Dataset ไม่มี mileage / condition / accidents → ทำให้ราคาแกว่งได้
        - Text parsing อาจพลาดบางรุ่นที่ format แปลก
        """)
    st.markdown('</div>', unsafe_allow_html=True)
