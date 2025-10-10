import streamlit as st
import joblib
import pandas as pd
import numpy as np

model = joblib.load('best_diabetes_classification_model.pkl')

st.title("Prédiction du risque diabète")

pregnancies = st.number_input("Nombre de grossesses", min_value=0, max_value=100, value=0)
glucose       = st.number_input("Glucose (mg/dL)", min_value=50, max_value=300, value=120)
blood_pressure= st.number_input("Pression artérielle (mmHg)", min_value=40, max_value=200, value=70)
skin_thickness= st.number_input("Épaisseur du pli cutané (mm)", min_value=5, max_value=100, value=20)
insulin       = st.number_input("Insuline (muU/mL)", min_value=0, max_value=900, value=80)
bmi           = st.number_input("BMI", min_value=10.0, max_value=70.0, value=32.0)
dpf           = st.number_input("DiabetesPedigreeFunction", min_value=0.05, max_value=2.5, value=0.5)
age           = st.number_input("Âge", min_value=10, max_value=120, value=30)

input_df = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [blood_pressure],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [dpf],
    'Age': [age]
})


if st.button("Prédire le risque"):
    cluster = model.predict(input_df)[0]
    risk_label = "risque élevé" if cluster == 0 else "risque faible"
    st.success(f"Résultat : **{risk_label}** (cluster {cluster})")