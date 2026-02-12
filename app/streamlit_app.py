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


# Style
apply_dark_style()

# Load model/meta 
pipe, meta = load_or_train()

# Sidebar 
retrain_clicked = render_sidebar(meta)
if retrain_clicked:
    load_or_train.clear()
    pipe, meta = train_and_save_in_cloud()
    st.success("Retrained! (ลองกด Rerun / Refresh หน้าเว็บอีกที)")

# Header
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

# Prepare data
df_raw = pd.read_csv(DATA_PATH)
df_feat = build_features(df_raw).dropna(subset=["Brand", "Model", "Property", "Year", "PowerCC", "Turbo", "Price"]).copy()

# Main layout
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

    # Currency settings 
    st.caption("Currency: Jordanian Dinar (JOD)")
    # อัตราแลกเปลี่ยนแบบประมาณ
    FX_JOD_TO_THB = 43.65

    # ตัวเลือกให้ผู้ใช้ปรับ rate เอง 
    FX_JOD_TO_THB = st.number_input(
        "Exchange rate (THB per 1 JOD)",
        min_value=1.0,
        max_value=200.0,
        value=float(FX_JOD_TO_THB),
        step=0.5,
        help="Approximate rate for quick conversion. อัตราจริงอาจเปลี่ยนแปลงได้"
    )

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

        # Convert to THB 
        pred_thb = pred * FX_JOD_TO_THB
        low_thb = low * FX_JOD_TO_THB
        high_thb = high * FX_JOD_TO_THB

        st.markdown(
            f"""
            <div style="font-size:2.1rem; font-weight:900;">{pred:,.0f} JOD</div>
            <div class="muted">Estimated range (±RMSE): {low:,.0f} – {high:,.0f} JOD</div>
            <div style="height:10px;"></div>
            <div style="font-size:1.35rem; font-weight:800;">≈ {pred_thb:,.0f} THB <span class="muted" style="font-weight:600;">(approx.)</span></div>
            <div class="muted">Approx range: {low_thb:,.0f} – {high_thb:,.0f} THB</div>
            """,
            unsafe_allow_html=True
        )

        st.caption(
            "Note: THB conversion uses an approximate exchange rate and may vary. "
            "Prediction reflects listed market price in dataset; may not include additional fees (registration/transfer/etc.)."
        )

    st.markdown('</div>', unsafe_allow_html=True)


# Tabs (NO matplotlib / NO seaborn)
tab1, tab2, tab3 = st.tabs(["📄 Dataset Preview", "📊 Insights", "ℹ️ About Model"])

with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📄 Dataset Preview")
    st.caption("Selected rows (sanity check) — แถวที่ match กับ Brand + Model + Property ที่เลือก")

    if subset is not None and len(subset) > 0:
        cols_show = [c for c in ["Model", "Brand", "Property", "Power", "Year", "PowerCC", "Turbo", "Price"] if c in subset.columns]
        st.dataframe(subset[cols_show].head(30), use_container_width=True)
        st.caption(f"Rows matched: {len(subset):,}")
    else:
        st.info("No exact row match for this selection (ยังปกติ — โมเดลสามารถทำนายจาก pattern รวมได้)")

    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Insights")
    st.caption("Quick insights (filtered by Brand if possible) — โหมดนี้ไม่ใช้ matplotlib เลย")

    # Filter by selected brand if possible
    brand_val = None
    try:
        brand_val = str(input_df.loc[0, "Brand"])
    except Exception:
        brand_val = None

    if brand_val and "Brand" in df_feat.columns:
        df_plot = df_feat[df_feat["Brand"] == brand_val].copy()
        if len(df_plot) < 30:
            df_plot = df_feat.copy()
            st.info(f"Brand '{brand_val}' มีแถวน้อย (<30) เลยโชว์ภาพรวมทั้ง dataset แทน")
        else:
            st.success(f"Showing insights for Brand = **{brand_val}** (rows: {len(df_plot):,})")
    else:
        df_plot = df_feat.copy()
        st.info("Showing insights for all brands")

    # Basic stats
    price_s = pd.to_numeric(df_plot.get("Price", pd.Series([], dtype=float)), errors="coerce").dropna()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df_plot):,}")
    c2.metric("Min", f"{price_s.min():,.0f} JOD" if len(price_s) else "-")
    c3.metric("Median", f"{price_s.median():,.0f} JOD" if len(price_s) else "-")
    c4.metric("Max", f"{price_s.max():,.0f} JOD" if len(price_s) else "-")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Price distribution (simple bins table + bar chart from counts)
    st.write("### Price distribution (bins)")
    if len(price_s) > 0:
        bins = st.slider("Bins", 10, 60, 30, 5, key="bins_no_mpl")
        binned = pd.cut(price_s, bins=bins)
        counts = binned.value_counts().sort_index()
        chart_df = pd.DataFrame({"count": counts.values}, index=counts.index.astype(str))

        st.bar_chart(chart_df, height=260)
        with st.expander("Show bin counts (table)"):
            st.dataframe(chart_df, use_container_width=True)
    else:
        st.warning("ไม่มีข้อมูล Price ที่ใช้ทำ distribution ได้")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Top brands by count (global, not filtered)
    st.write("### Top brands (count)")
    if "Brand" in df_feat.columns:
        topn = st.slider("Top N brands", 5, 25, 10, 5, key="topn_brand")
        vc = df_feat["Brand"].value_counts().head(topn)
        st.bar_chart(vc)
        with st.expander("Show top brands (table)"):
            st.dataframe(vc.rename("count").reset_index().rename(columns={"index": "Brand"}), use_container_width=True)
    else:
        st.info("ไม่พบคอลัมน์ Brand")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Year vs Price (use Streamlit scatter_chart)
    st.write("### Year vs Price (sample)")
    if "Year" in df_plot.columns and "Price" in df_plot.columns:
        tmp = df_plot[["Year", "Price"]].dropna().copy()
        if len(tmp) > 0:
            tmp = tmp.sample(min(800, len(tmp)), random_state=42)
            st.scatter_chart(tmp, x="Year", y="Price", height=320)
        else:
            st.info("ไม่มีข้อมูล Year/Price พอจะ plot ได้")
    else:
        st.info("ไม่พบคอลัมน์ Year หรือ Price")

    with st.expander("Show sample rows (sanity check)"):
        cols_show = [c for c in ["Model", "Brand", "Property", "Power", "Year", "PowerCC", "Turbo", "Price"] if c in df_plot.columns]
        st.dataframe(df_plot[cols_show].head(20), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("ℹ️ About Model")

    r2 = float(meta["metrics"]["r2"])
    mae = float(meta["metrics"]["mae"])
    rmse = float(meta["metrics"]["rmse"])

    st.markdown(
        f"""
        **Pipeline**
        - Feature engineering: Brand/Year from `Model`, PowerCC/Turbo from `Power`
        - Preprocess: StandardScaler (numeric) + OneHotEncoder (categorical)
        - Model: Ridge Regression (Linear Regression family)

        **Metrics**
        - **R² = {r2:.3f}** → ใกล้ **1** ดี (โมเดลอธิบายความแปรปรวนของราคาได้มากขึ้น)
        - **MAE = {mae:,.0f}** → ใกล้ **0** ดี (ค่า error เฉลี่ยน้อยลง)
        - **RMSE = {rmse:,.0f}** → ใกล้ **0** ดี (ลงโทษ error ก้อนใหญ่แรงกว่า MAE)

        **Limitations**
        - Dataset ไม่มี mileage / condition / accidents → ทำให้ราคาแกว่ง
        - การ parse ข้อความจาก `Model`/`Power` อาจพลาดบาง format ที่แปลก
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
