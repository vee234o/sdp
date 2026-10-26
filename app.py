import streamlit as st
import pandas as pd
import joblib

# load saved model and scaler
rf_model = joblib.load('sleep_disorder_rf_model.pkl')
scaler = joblib.load('feature_scaler.pkl')

st.set_page_config(page_title="Sleep Disorder Prediction System", layout="centered")

st.title("Sleep Disorder Prediction System")
st.write("Enter patient vitals to predict sleep disorder risk.")

# create form for user inputs
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        sleep_duration = st.number_input("Sleep Duration (hours)", min_value=3.0, max_value=12.0, value=7.2, step=0.1)
        quality_of_sleep = st.number_input("Quality of Sleep (1 to 10)", min_value=1, max_value=10, value=7)
        physical_activity = st.number_input("Physical Activity (minutes per day)", min_value=0, max_value=180, value=60)
        stress_level = st.number_input("Stress Level (1 to 10)", min_value=1, max_value=10, value=5)

    with col2:
        bmi_category = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese"])
        systolic_bp = st.number_input("Systolic Blood Pressure", min_value=80, max_value=200, value=120)
        diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=50, max_value=130, value=80)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=150, value=70)
        daily_steps = st.number_input("Daily Steps", min_value=500, max_value=30000, value=7000, step=100)

    submit = st.form_submit_button("Predict Risk")

if submit:
    # convert text selections to numbers matching the training data
    gender_val = 0 if gender == "Male" else 1
    bmi_map = {"Normal": 0, "Overweight": 1, "Obese": 2}
    bmi_val = bmi_map[bmi_category]

    # build dataframe in the exact order used during training
    input_df = pd.DataFrame([[
        gender_val, age, sleep_duration, quality_of_sleep,
        physical_activity, stress_level, bmi_val,
        systolic_bp, diastolic_bp, heart_rate, daily_steps
    ]], columns=[
        'Gender', 'Age', 'Sleep Duration', 'Quality of Sleep',
        'Physical Activity Level', 'Stress Level', 'BMI Category',
        'Systolic_BP', 'Diastolic_BP', 'Heart Rate', 'Daily Steps'
    ])

    # scale the continuous columns
    continuous_cols = [
        'Age', 'Sleep Duration', 'Physical Activity Level',
        'Stress Level', 'Systolic_BP', 'Diastolic_BP',
        'Heart Rate', 'Daily Steps'
    ]
    input_df[continuous_cols] = scaler.transform(input_df[continuous_cols])

    # run prediction and get probabilities
    prediction = rf_model.predict(input_df)[0]
    probabilities = rf_model.predict_proba(input_df)[0]

    disorder_labels = {0: "None", 1: "Insomnia", 2: "Sleep Apnea"}
    predicted_label = disorder_labels[prediction]

    st.subheader("Prediction Result")
    st.write(f"Predicted Condition: **{predicted_label}**")

    st.write("Probability Breakdown:")
    prob_df = pd.DataFrame({
        "Condition": ["None", "Insomnia", "Sleep Apnea"],
        "Probability": [f"{p * 100:.1f}%" for p in probabilities]
    })
    st.table(prob_df)