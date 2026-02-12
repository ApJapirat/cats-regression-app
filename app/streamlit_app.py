# app/streamlit_app.py
import os
import sys
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath("."))

from src.features import build_features
from src.model import load_or_train, retrain
from app.ui import render_sidebar, render_inputs
from app.style import apply_dark_style


st.set_page_config(
    page_title="Car Price Predictor (Jordan)",
    page_icon="🚗",
    layout="wide"
)

DATA_PATH = "data/car_prices_jordan.csv"


# Style 
apply_dark_style()

# Load model/meta 
pipe, meta = load_or_train()

# Sidebar
retrain_clicked = render_sidebar(meta)
if retrain_clicked:
    pipe, meta = retrain()
    st.success("Retrained successfully! (ถ้าไม่เปลี่ยน ลองกด Rerun/Refresh)")

# Header
m = meta.get("metrics", {})
st.markdown(
    f"""
    <div class="card glow">
      <div style="display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap;">
        <div>
          <div style="font-size: 2.0rem; font-weight: 800;">🚗 Car Price Predictor <span class="muted">(Jordan)</span></div>
          <div class="muted">Predict used-car price (JOD) using linked inputs: Brand → Model → Property (+ Year/PowerCC/Turbo)</div>
        </div>
        <div style="text-align:right;">
          <div class="tiny">Model</div>
          <div style="font-weight:700;">Linear Regression (Ridge)</div>
          <div class="tiny">R²: {float(m.get('r2',0)):.3f} • RMSE: {float(m.get('rmse',0)):.0f}</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("")

# Prepare dataset
df_raw = pd.read_csv(DATA_PATH)
df_feat = build_features(df_raw).dropna(
    subset=[c for c in ["Brand", "Model", "Property", "Year", "PowerCC", "Turbo", "Price"] if c in build_features(df_raw).columns]
).copy()

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
    st.caption("Currency: Jordanian Dinar (JOD)")

    # FX conversion (approx)
    default_fx = 43.65  # THB per 1 JOD (approx)
    fx = st.number_input(
        "Exchange rate (THB per 1 JOD)",
        min_value=1.0, max_value=200.0,
        value=float(default_fx),
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
        rmse = float(meta.get("metrics", {}).get("rmse", 0.0))
        low, high = max(0.0, pred - rmse), pred + rmse

        pred_thb = pred * fx
        low_thb = low * fx
        high_thb = high * fx

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
            "Prediction reflects listed market price in dataset; may not include additional fees (tax/registration/transfer/etc.)."
        )

    st.markdown('</div>', unsafe_allow_html=True)

# Tabs 
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
        st.info("No exact row match (ยังปกติ — โมเดลสามารถทำนายจาก pattern รวมได้)")

    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Insights")
    st.caption("Quick insights — ไม่ใช้ matplotlib/seaborn (ลดปัญหา deploy)")

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

    price_s = pd.to_numeric(df_plot.get("Price", pd.Series([], dtype=float)), errors="coerce").dropna()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df_plot):,}")
    c2.metric("Min", f"{price_s.min():,.0f} JOD" if len(price_s) else "-")
    c3.metric("Median", f"{price_s.median():,.0f} JOD" if len(price_s) else "-")
    c4.metric("Max", f"{price_s.max():,.0f} JOD" if len(price_s) else "-")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

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

    with st.expander("Show sample rows"):
        cols_show = [c for c in ["Model", "Brand", "Property", "Power", "Year", "PowerCC", "Turbo", "Price"] if c in df_plot.columns]
        st.dataframe(df_plot[cols_show].head(20), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("ℹ️ About Model")

    r2 = float(meta.get("metrics", {}).get("r2", 0.0))
    mae = float(meta.get("metrics", {}).get("mae", 0.0))
    rmse = float(meta.get("metrics", {}).get("rmse", 0.0))
    use_year = bool(meta.get("use_year", True))

    st.markdown(
        f"""
        **Pipeline**
        - Feature engineering: Brand/Year from `Model`, PowerCC/Turbo from `Power`
        - Preprocess: StandardScaler (numeric) + OneHotEncoder (categorical)
        - Model: Ridge Regression (Linear Regression family)

        **Features used**
        - use_year = **{use_year}**
        - features = `{meta.get("features", [])}`

        **Metrics**
        - **R² = {r2:.3f}** → ใกล้ **1** ดี
        - **MAE = {mae:,.0f}** → ใกล้ **0** ดี
        - **RMSE = {rmse:,.0f}** → ใกล้ **0** ดี (ลงโทษ error ใหญ่แรงกว่า MAE)

        **Limitations**
        - Dataset ไม่มี mileage / condition / accidents → ทำให้ราคาแกว่ง
        - การ parse ข้อความจาก `Model`/`Power` อาจพลาดบาง format ที่แปลก
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
