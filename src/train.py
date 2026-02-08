# src/train.py
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

#Load dataset
DATA_PATH = os.getenv(
    "DATA_PATH",
    "data/cats_dataset.csv"
)

df = pd.read_csv(DATA_PATH)

# ทำชื่อคอลัมน์ให้อ่านง่าย และเผื่อไม่ตรงกับไฟล์
col_map = {
    "Age (Years)": "Age",
    "Weight (kg)": "Weight"
}
df = df.rename(columns=col_map)

# ลบช่องว่างและจัดรูปแบบข้อความในคอลัมน์หมวดหมู่
for c in ["Breed", "Color", "Gender"]:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip()

# เลือก features (<=5) และ target
features = ["Age", "Breed", "Color", "Gender"]
target = "Weight"

# ตัดแถวที่ค่าว่างในคอลัมน์สำคัญ
df = df.dropna(subset=features + [target]).copy()

X = df[features]
y = df[target].astype(float)

#Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#Preprocess and Model
#- ใช้ OrdinalEncoder เพื่อให้ "จำนวน feature หลังแปลง" ยังน้อย 
num_cols = ["Age"]
cat_cols = ["Breed", "Color", "Gender"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
    ],
    remainder="drop"
)

model = LinearRegression()

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model)
])

#Train
pipe.fit(X_train, y_train)

#Evaluate
y_pred = pipe.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("=== Evaluation on Test Set ===")
print(f"R^2  : {r2:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")


#Save model and metadata
os.makedirs("models", exist_ok=True)
joblib.dump(pipe, "models/cat_weight_model.joblib")

# เก็บค่าที่ใช้ทำ dropdown บนหน้าเว็บ (list ของ breed/color/gender)
metadata = {
    "features": features,
    "target": target,
    "breed_options": sorted(df["Breed"].unique().tolist()),
    "color_options": sorted(df["Color"].unique().tolist()),
    "gender_options": sorted(df["Gender"].unique().tolist()),
    "metrics": {"r2": float(r2), "mae": float(mae), "rmse": float(rmse)}
}

with open("models/metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("\nSaved:")
print("- models/cat_weight_model.joblib")
print("- models/metadata.json")
