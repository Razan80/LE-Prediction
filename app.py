%%writefile le_predictor_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

@st.cache_data
def baseline_le(age, sex):
    if sex.lower() == 'm':
        return max(85 - age * 0.15, 50)
    else:
        return max(87 - age * 0.13, 52)

def calculate_hrs(bca_data, health_data):
    height_m = bca_data['height_m']
    weight = bca_data['weight']
    bmi = weight / (height_m ** 2)
    hrs = 0
    hrs += max(0, (bmi - 22.5) / 10) * 20
    hrs += max(0, bca_data['vfl'] - 7) * 15
    ideal_bf = 15 if health_data['sex'].lower() == 'm' else 23
    hrs += max(0, (bca_data['bf_pct'] - ideal_bf) / 5) * 10
    hrs += max(0, (32 - bca_data['smm_kg']) / 5) * 8
    hrs += max(0, (1500 - bca_data['bmr']) / 200) * 5
    if health_data.get('smoking', False): hrs += 15
    if health_data.get('bp', 120) > 140: hrs += 10
    if health_data.get('glucose', 90) > 100: hrs += 12
    if health_data.get('family_cvd', False): hrs += 8
    hrs = np.clip(hrs, 0, 100)
    return hrs, bmi

def heuristic_predict(bca_data, health_data):
    base_le = baseline_le(health_data['age'], health_data['sex'])
    hrs, bmi = calculate_hrs(bca_data, health_data)
    pred_le = max(40, base_le - hrs * 0.15)
    tips = []
    if bmi > 25: tips.append("💪 Lose weight: +2–5 yrs!")
    if bca_data['vfl'] > 7: tips.append("🔥 Cavitation: +3 yrs!")
    if hrs > 50: tips.append("🌿 Wellness program: +5+ yrs!")
    return {'predicted_le': round(pred_le, 1), 'hrs': round(hrs, 1), 'bmi': round(bmi, 1), 'tips': tips}

st.set_page_config(page_title="Healthy Home LE Predictor", layout="wide")
st.title("🩺 Healthy Home Life Expectancy Predictor")
st.markdown("Wellness insights only — not medical advice.")

st.write("App is running correctly 🚀")
