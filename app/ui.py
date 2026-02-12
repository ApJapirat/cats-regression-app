# app/ui.py
import pandas as pd
import streamlit as st


def render_sidebar(meta: dict) -> bool:
    with st.sidebar:
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.subheader("Model Info")
        st.write("Target:", meta.get("target", "-"))
        st.write("Features:", ", ".join(meta.get("features", [])))

        st.write("Test Metrics (hold-out test set):")
        m = meta.get("metrics", {})
        st.metric("R²", f"{float(m.get('r2', 0.0)):.3f}")
        st.metric("MAE", f"{float(m.get('mae', 0.0)):.0f}")
        st.metric("RMSE", f"{float(m.get('rmse', 0.0)):.0f}")

        st.caption("R² ใกล้ 1 ดี, MAE/RMSE ใกล้ 0 ดี")

        retrain_clicked = st.button("🔁 Retrain model (cloud)", use_container_width=True)

    return retrain_clicked


def render_inputs(meta: dict, df_feat: pd.DataFrame):
    """
    Linked inputs:
    Brand -> Model -> Property -> (Year/PowerCC/Turbo constrained if matched rows exist)
    Return:
    - input_df (1 row) with columns required by model features
    - subset (matched rows for sanity check)
    """
    opts = meta.get("options", {})
    brands = opts.get("brands", [])
    brand_to_models = opts.get("brand_to_models", {})
    brand_model_to_props = opts.get("brand_model_to_props", {})

    st.subheader("Input Features")

    if not brands:
        st.error("No brand options found in metadata. Please retrain.")
        return pd.DataFrame([{}]), df_feat.iloc[0:0]

    # Brand 
    brand = st.selectbox("Brand", options=brands, index=0)

    # Model 
    models = brand_to_models.get(brand, [])
    if not models:
        models = sorted(df_feat.loc[df_feat["Brand"] == brand, "Model"].dropna().unique().tolist())

    model = st.selectbox("Model", options=models, index=0 if models else 0)

    # Property  
    key = f"{brand}|||{model}"
    props = brand_model_to_props.get(key, [])
    if not props:
        props = sorted(df_feat.loc[(df_feat["Brand"] == brand) & (df_feat["Model"] == model), "Property"].dropna().unique().tolist())

    prop = st.selectbox("Property (transmission)", options=props if props else ["-"], index=0)

    # Matched subset rows for constraint + preview
    subset = df_feat.loc[
        (df_feat["Brand"] == brand) &
        (df_feat["Model"] == model) &
        (df_feat["Property"] == prop)
    ].copy()

    # Constrain Year/PowerCC/Turbo if subset exists 
    if len(subset) > 0:
        year_options = sorted(subset["Year"].dropna().unique().astype(int).tolist())
        power_options = sorted(subset["PowerCC"].dropna().unique().astype(float).tolist())
        turbo_options = sorted(subset["Turbo"].dropna().unique().astype(int).tolist())

        year = st.selectbox("Year", options=year_options, index=0)
        power_cc = st.selectbox("Power (CC)", options=power_options, index=0)
        turbo = st.selectbox("Turbo", options=turbo_options, format_func=lambda x: "Yes" if int(x) == 1 else "No")
        st.caption(f"🔗 Linked inputs: found {len(subset)} row(s) for this Brand+Model+Property, so Year/PowerCC/Turbo are constrained.")
    else:
        # fallback: allow manual input
        year = st.number_input("Year", min_value=1990, max_value=2026, value=2020, step=1)
        power_cc = st.number_input("Power (CC)", min_value=0.0, max_value=8000.0, value=1500.0, step=50.0)
        turbo = st.selectbox("Turbo", options=[0, 1], format_func=lambda x: "Yes" if int(x) == 1 else "No")
        st.caption("No exact row match — still ok. Model can generalize from patterns.")

    # Build input_df matching model feature list
    use_year = bool(meta.get("use_year", True))
    if use_year:
        input_df = pd.DataFrame([{
            "Brand": brand,
            "Model": model,
            "Property": prop,
            "Year": int(year),
            "PowerCC": float(power_cc),
            "Turbo": int(turbo),
        }])
    else:
        input_df = pd.DataFrame([{
            "Brand": brand,
            "Model": model,
            "Property": prop,
            "PowerCC": float(power_cc),
            "Turbo": int(turbo),
        }])

    return input_df, subset
