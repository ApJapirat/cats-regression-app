# src/train.py
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.features import build_features


DATA_PATH = os.getenv("DATA_PATH", "data/car_prices_jordan.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "models/car_price_model.joblib")
META_PATH = os.getenv("META_PATH", "models/metadata.json")


def main():
    # Load
    df_raw = pd.read_csv(DATA_PATH)

    # Feature engineering
    df = build_features(df_raw)

    features = ["Brand", "Model", "Property", "Year", "PowerCC", "Turbo"]
    target = "Price"

    # drop rows ที่ใช้ไม่ได้
    df = df.dropna(subset=features + [target]).copy()

    X = df[features]
    y = df[target].astype(float)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocess
    num_cols = ["Year", "PowerCC", "Turbo"]
    cat_cols = ["Brand", "Model", "Property"]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    # Ridge = linear regression family (ยังเป็น linear model)
    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", Ridge(alpha=1.0)),
    ])

    # Train
    pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    print("=== Evaluation on Test Set ===")
    print(f"R^2  : {r2:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")

    # Save
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)

    # ทำ dropdown แบบ dependent: Brand -> Model -> Property
    brands = sorted(df["Brand"].dropna().unique().tolist())

    brand_to_models = {}
    brand_model_to_props = {}

    for b in brands:
        models = sorted(df.loc[df["Brand"] == b, "Model"].dropna().unique().tolist())
        brand_to_models[b] = models
        for m in models:
            props = sorted(df.loc[(df["Brand"] == b) & (df["Model"] == m), "Property"].dropna().unique().tolist())
            brand_model_to_props[f"{b}|||{m}"] = props

    metadata = {
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
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nSaved:")
    print(f"- {MODEL_PATH}")
    print(f"- {META_PATH}")


if __name__ == "__main__":
    main()
