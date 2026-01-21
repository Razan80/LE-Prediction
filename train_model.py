import pandas as pd
from sklearn. ensemble import RandomForestRegressor
import joblib

# Load training data
df = pd.read_csv("training_data.csv")

X = df[
    [
        "age",
        "sex_male",
        "height_m",
        "weight",
        "bf_pct",
        "vfl",
        "smm_kg",
        "bmr",
        "smoking",
        "bp",
        "glucose",
        "family_cvd",
    ]
]

y = df["le_remaining"]

# Train model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

# Save model
joblib.dump(model, "le_model.pkl")

print("✅ Model trained successfully")
print("R² score:", round(model.score(X, y), 3))
