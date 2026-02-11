# app/style.py
import streamlit as st

def apply_dark_style():
    st.markdown(
        """
        <style>
        /* --------- Base --------- */
        .block-container { padding-top: 2rem; }
        h1, h2, h3 { letter-spacing: 0.3px; }

        /* --------- Card --------- */
        .card {
          background: rgba(17, 24, 39, 0.85);
          border: 1px solid rgba(255, 255, 255, 0.08);
          border-radius: 18px;
          padding: 16px 18px;
          box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        }

        .glow {
          border: 1px solid rgba(255, 75, 75, 0.35);
          box-shadow: 0 0 0 1px rgba(255, 75, 75, 0.18),
                      0 12px 40px rgba(255, 75, 75, 0.10);
        }

        .muted { color: rgba(229,231,235,0.75); font-size: 0.95rem; }
        .tiny  { color: rgba(229,231,235,0.65); font-size: 0.85rem; }

        .divider {
          height: 1px;
          background: rgba(255,255,255,0.08);
          margin: 10px 0 14px 0;
        }

        /* --------- Metric style tweak --------- */
        [data-testid="stMetric"] {
          background: rgba(17, 24, 39, 0.75);
          border: 1px solid rgba(255, 255, 255, 0.06);
          padding: 10px 12px;
          border-radius: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
