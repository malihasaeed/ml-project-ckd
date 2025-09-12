# app.py - Streamlit CKD Prediction App

import streamlit as st
import pandas as pd
import joblib

# 1. Page configuration
st.set_page_config(
    page_title="Chronic Kidney Disease Prediction",
    page_icon="💉",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. App title
st.title("Chronic Kidney Disease Prediction")
st.write("Enter patient data to predict the likelihood of CKD.")

# 3. Load the trained model
model = joblib.load("best_model.pkl")  # Make sure your saved model is in the same folder

# 4. User input
st.sidebar.header("Patient Information")

age = st.sidebar.number_input("Age (years)", min_value=1, max_value=120, value=30)
gender = st.sidebar.selectbox("Gender", ["male", "female"])
blood_pressure = st.sidebar.number_input("Systolic BP (mmHg)", min_value=50, max_value=200, value=120)
specific_gravity = st.sidebar.selectbox("Specific Gravity", ["1.010", "1.015", "1.020", "1.025", "1.030"])
albumin = st.sidebar.selectbox("Albumin (0-5)", [0, 1, 2, 3, 4, 5])
sugar = st.sidebar.selectbox("Sugar (0-5)", [0, 1, 2, 3, 4, 5])
pus_cell = st.sidebar.selectbox("Pus Cell", ["normal", "abnormal"])
pus_cell_clumps = st.sidebar.selectbox("Pus Cell Clumps", ["absent", "present"])
bacteria = st.sidebar.selectbox("Bacteria in urine", ["absent", "present"])
blood_glucose_random = st.sidebar.number_input("Random Blood Glucose (mg/dL)", min_value=50, max_value=500, value=100)
blood_urea = st.sidebar.number_input("Blood Urea (mg/dL)", min_value=5, max_value=200, value=20)
serum_creatinine = st.sidebar.number_input("Serum Creatinine (mg/dL)", min_value=0.1, max_value=20.0, value=1.0)
sodium = st.sidebar.number_input("Sodium (mEq/L)", min_value=120, max_value=160, value=140)
potassium = st.sidebar.number_input("Potassium (mEq/L)", min_value=2, max_value=10, value=4)
hemoglobin = st.sidebar.number_input("Hemoglobin (g/dL)", min_value=5, max_value=20, value=14)
packed_cell_volume = st.sidebar.number_input("Packed Cell Volume (%)", min_value=10, max_value=60, value=40)
white_blood_cell_count = st.sidebar.number_input("WBC Count (cells/cumm)", min_value=1000, max_value=20000, value=7000)
red_blood_cell_count = st.sidebar.number_input("RBC Count (millions/cumm)", min_value=1, max_value=10, value=5)
hypertension = st.sidebar.selectbox("Hypertension", ["no", "yes"])
diabetes_mellitus = st.sidebar.selectbox("Diabetes Mellitus", ["no", "yes"])
coronary_artery_disease = st.sidebar.selectbox("Coronary Artery Disease", ["no", "yes"])
appetite = st.sidebar.selectbox("Appetite", ["good", "poor"])
anemia = st.sidebar.selectbox("Anemia", ["no", "yes"])
pedal_edema = st.sidebar.selectbox("Pedal Edema", ["no", "yes"])

# 5. Prepare input dataframe
input_data = pd.DataFrame({
    'age': [age],
    'gender': [gender],
    'blood_pressure': [blood_pressure],
    'specific_gravity': [specific_gravity],
    'albumin': [albumin],
    'sugar': [sugar],
    'pus_cell': [pus_cell],
    'pus_cell_clumps': [pus_cell_clumps],
    'bacteria': [bacteria],
    'blood_glucose_random': [blood_glucose_random],
    'blood_urea': [blood_urea],
    'serum_creatinine': [serum_creatinine],
    'sodium': [sodium],
    'potassium': [potassium],
    'hemoglobin': [hemoglobin],
    'packed_cell_volume': [packed_cell_volume],
    'white_blood_cell_count': [white_blood_cell_count],
    'red_blood_cell_count': [red_blood_cell_count],
    'hypertension': [hypertension],
    'diabetes_mellitus': [diabetes_mellitus],
    'coronary_artery_disease': [coronary_artery_disease],
    'appetite': [appetite],
    'anemia': [anemia],
    'pedal_edema': [pedal_edema]
})

# 6. Prediction button
if st.button("Predict CKD"):
    # Encode categorical features (same as in training)
    mapping_gender = {"male": 1, "female": 0}
    mapping_binary = {"yes": 1, "no": 0}
    mapping_pus_cell = {"normal": 0, "abnormal": 1}
    mapping_pus_cell_clumps = {"absent": 0, "present": 1}
    mapping_bacteria = {"absent": 0, "present": 1}
    mapping_appetite = {"good": 0, "poor": 1}
    
    # Apply mappings
    input_data['gender'] = input_data['gender'].map(mapping_gender)
    input_data['hypertension'] = input_data['hypertension'].map(mapping_binary)
    input_data['diabetes_mellitus'] = input_data['diabetes_mellitus'].map(mapping_binary)
    input_data['coronary_artery_disease'] = input_data['coronary_artery_disease'].map(mapping_binary)
    input_data['anemia'] = input_data['anemia'].map(mapping_binary)
    input_data['pedal_edema'] = input_data['pedal_edema'].map(mapping_binary)
    input_data['pus_cell'] = input_data['pus_cell'].map(mapping_pus_cell)
    input_data['pus_cell_clumps'] = input_data['pus_cell_clumps'].map(mapping_pus_cell_clumps)
    input_data['bacteria'] = input_data['bacteria'].map(mapping_bacteria)
    input_data['appetite'] = input_data['appetite'].map(mapping_appetite)
    
    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    
    # Display results
    if prediction[0] == 1:
        st.error(f"✅ The patient is predicted to HAVE CKD.\nPrediction probability: {prediction_proba}")
    else:
        st.success(f"✅ The patient is predicted NOT to have CKD.\nPrediction probability: {prediction_proba}")

