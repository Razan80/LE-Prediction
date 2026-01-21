import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------ CORE LOGIC ------------------
@st.cache_resource
def load_ml_model():
    try:
        model = joblib.load("le_model.pkl")
        return model
    except:
        return None

@st.cache_data
def baseline_le(age, sex):
    if sex.lower() == "m":
        return max(85 - age * 0.15, 50)
    else:
        return max(87 - age * 0.13, 52)

def calculate_hrs(bca_data, health_data):
    height_m = bca_data["height_m"]
    weight = bca_data["weight"]
    bmi = weight / (height_m ** 2)

    hrs = 0
    hrs += max(0, (bmi - 22.5) / 10) * 20
    hrs += max(0, bca_data["vfl"] - 7) * 15

    ideal_bf = 15 if health_data["sex"].lower() == "m" else 23
    hrs += max(0, (bca_data["bf_pct"] - ideal_bf) / 5) * 10
    hrs += max(0, (32 - bca_data["smm_kg"]) / 5) * 8
    hrs += max(0, (1500 - bca_data["bmr"]) / 200) * 5

    if health_data.get("smoking"):
        hrs += 15
    if health_data.get("bp", 120) > 140:
        hrs += 10
    if health_data.get("glucose", 90) > 100:
        hrs += 12
    if health_data.get("family_cvd"):
        hrs += 8

    hrs = np.clip(hrs, 0, 100)
    return round(hrs, 1), round(bmi, 1)

def heuristic_predict(bca_data, health_data):
    base = baseline_le(health_data["age"], health_data["sex"])
    hrs, bmi = calculate_hrs(bca_data, health_data)
    pred = max(40, base - hrs * 0.15)

    tips = []
    if bmi > 25:
        tips.append("💪 Weight reduction may improve longevity")
    if bca_data["vfl"] > 7:
        tips.append("🔥 Reducing visceral fat improves metabolic health")
    if hrs > 50:
        tips.append("🌿 Structured wellness program strongly recommended")

    return round(pred, 1), hrs, bmi, tips

# ------------------ UI ------------------
st.sidebar.header("⚙️ Prediction Mode")
use_ml = st.sidebar.toggle(
    "Use ML Prediction (experimental)",
    value=False,
    help="Uses trained model if available. Heuristic is recommended."
)

st.set_page_config(
    page_title="Healthy Home Life Expectancy Predictor",
    layout="wide"
)

st.title("🩺 Healthy Home Life Expectancy Predictor")
st.warning(
    "This tool provides wellness insights only. "
    "It is NOT a medical diagnosis or a life expectancy guarantee."
)

# Sidebar Inputs
st.sidebar.header("📊 Client Information")

age = st.sidebar.number_input("Age", 18, 100, 45)
sex = st.sidebar.selectbox("Sex", ["M", "F"])
height_cm = st.sidebar.number_input("Height (cm)", 140, 220, 165)
weight = st.sidebar.number_input("Weight (kg)", 40, 200, 70)

st.sidebar.header("InBody BCA")
bf_pct = st.sidebar.number_input("Body Fat %", 5, 60, 25)
vfl = st.sidebar.number_input("Visceral Fat Level", 1, 20, 8)
smm_kg = st.sidebar.number_input("Skeletal Muscle Mass (kg)", 20, 50, 30)
bmr = st.sidebar.number_input("BMR (kcal)", 1000, 2500, 1500)

st.sidebar.header("Health History")
smoking = st.sidebar.checkbox("Smoker")
bp = st.sidebar.number_input("Systolic BP", 90, 200, 120)
glucose = st.sidebar.number_input("Fasting Glucose", 70, 300, 90)
family_cvd = st.sidebar.checkbox("Family History of Heart Disease")

if st.button("🔮 Predict Life Expectancy", type="primary"):
    bca_data = {
        "height_m": height_cm / 100,
        "weight": weight,
        "bf_pct": bf_pct,
        "vfl": vfl,
        "smm_kg": smm_kg,
        "bmr": bmr,
    }

    health_data = {
        "age": age,
        "sex": sex,
        "smoking": smoking,
        "bp": bp,
        "glucose": glucose,
        "family_cvd": family_cvd,
    }

    # Heuristic prediction (always available)
    le, hrs, bmi, tips = heuristic_predict(bca_data, health_data)

    final_le = le
    method_used = "Heuristic Model"

    # Optional ML prediction
    if use_ml:
        model = load_ml_model()
        if model is not None:
            features = [[
                age,
                1 if sex == "M" else 0,
                bca_data["height_m"],
                weight,
                bf_pct,
                vfl,
                smm_kg,
                bmr,
                1 if smoking else 0,
                bp,
                glucose,
                1 if family_cvd else 0,
            ]]
            final_le = round(model.predict(features)[0], 1)
            method_used = "ML Model (Experimental)"
        else:
            st.info("ML model not found. Using heuristic prediction.")

    # Display results
    col1, col2, col3 = st.columns(3)
    col1.metric("Estimated LE", f"{final_le} years")
    col2.metric("BMI", bmi)
    col3.metric("Health Risk Index", f"{hrs} / 100")

    st.caption(f"Prediction method: {method_used}")

    st.subheader("💡 Wellness Insights")
    for tip in tips:
        st.write(tip)

    bca_data = {
        "height_m": height_cm / 100,
        "weight": weight,
        "bf_pct": bf_pct,
        "vfl": vfl,
        "smm_kg": smm_kg,
        "bmr": bmr,
    }

    health_data = {
        "age": age,
        "sex": sex,
        "smoking": smoking,
        "bp": bp,
        "glucose": glucose,
        "family_cvd": family_cvd,
    }

    le, hrs, bmi, tips = heuristic_predict(bca_data, health_data)

    col1, col2, col3 = st.columns(3)
    col1.metric("Estimated LE", f"{le} years")
    col2.metric("BMI", bmi)
    col3.metric("Health Risk Index", f"{hrs} / 100")

    st.subheader("💡 Wellness Insights")
    for tip in tips:
        st.write(tip)

st.sidebar.markdown("---")
st.sidebar.info("Built for Healthy Home 🇳🇵 | Wellness Analytics v1.0")

