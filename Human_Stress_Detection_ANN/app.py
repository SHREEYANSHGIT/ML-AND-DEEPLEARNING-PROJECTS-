import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import pickle

# Load model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = load_model(os.path.join(BASE_DIR, "stress_model.h5"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))

st.title("🧠 Human Stress Detection System")

st.write("Fill your lifestyle details below")

# ---------------- INPUTS ---------------- #

Age = st.number_input("Age",14,100,30)

Sleep_Duration = st.slider("Sleep Duration (hours)", 0.0, 12.0, 7.0)
Sleep_Quality = st.slider("Sleep Quality (1-10)", 1, 10, 5)

Physical_Activity = st.slider("Physical Activity (hours/day)", 0.0, 5.0, 1.0)
Screen_Time = st.slider("Screen Time (hours/day)", 0.0, 10.0, 4.0)

Caffeine_Intake = st.selectbox("Caffeine Intake (cups/day)", [0,1,2,3,4])

Alcohol = st.selectbox("Alcohol Consumption", ["No","Yes"])
Smoking = st.selectbox("Smoking Habit", ["No","Yes"])

Work_Hours = st.slider("Work Hours/day", 0.0, 16.0, 8.0)
Travel_Time = st.slider("Travel Time (hours)", 0.0, 5.0, 1.0)

Social_Interactions = st.slider("Social Interactions/day", 0.0, 10.0, 5.0)

Meditation = st.selectbox("Meditation Practice", ["No","Yes"])

Blood_Pressure = st.number_input("Blood Pressure", 90.0, 200.0, 120.0)
Cholesterol_Level = st.number_input("Cholesterol Level", 100.0, 300.0, 180.0)
Blood_Sugar_Level = st.number_input("Blood Sugar Level", 70.0, 200.0, 90.0)

Gender = st.selectbox("Gender", ["Male","Female"])

Status = st.selectbox("Marital Status", ["Single","Married"])

Wake_Up_Time_Hour = st.slider("Wake Up Time Hour", 0, 23, 7)
Bed_Time_Hour = st.slider("Bed Time Hour", 0, 23, 22)

# ---------------- ENCODING ---------------- #

Alcohol_Intake = 1 if Alcohol == "Yes" else 0
Smoking_Habit = 1 if Smoking == "Yes" else 0
Meditation_Practice = 1 if Meditation == "Yes" else 0

Male = 1 if Gender == "Male" else 0

Married = 1 if Status == "Married" else 0
Single = 1 if Status == "Single" else 0

# ---------------- PREDICTION ---------------- #

if st.button("Predict Stress Level"):

    features = np.array([[Age,
                          Sleep_Duration,
                          Sleep_Quality,
                          Physical_Activity,
                          Screen_Time,
                          Caffeine_Intake,
                          Alcohol_Intake,
                          Smoking_Habit,
                          Work_Hours,
                          Travel_Time,
                          Social_Interactions,
                          Meditation_Practice,
                          Blood_Pressure,
                          Cholesterol_Level,
                          Blood_Sugar_Level,
                          Male,
                          Married,
                          Single,
                          Wake_Up_Time_Hour,
                          Bed_Time_Hour]])

    features = scaler.transform(features)

    prediction = model.predict(features)

    if prediction[0][0] >= 0.5:
        st.error("⚠ Medium - High Stress Detected")
    else:
        st.success("✅ Low Stress Level")

    # ---------- FOOTER ----------

st.markdown(
    """
    <style>
    /* Add bottom padding so footer doesn't overlap chat input */
    .block-container {
        padding-bottom: 80px;
    }

    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: rgba(0,0,0,0);
        color: #B3B3B3;
        text-align: center;
        font-size: 16px;
        padding: 10px;
        z-index: 100;
    }

    /* Blue links */
    .footer a {
        color: #1DA1F2;   /* Blue */
        text-decoration: none;
        margin: 0 8px;
        font-weight: 500;
    }

    .footer a:hover {
        color: #0A66C2;   /* Darker blue on hover */
        text-decoration: underline;
    }
    </style>

    <div class="footer">
        © 2025 <b>Developed by Shreeyansh Asati</b> |
        <a href="https://www.linkedin.com/in/shreeyansh-asati-18shreey/" target="_blank">
            🔗 LinkedIn
        </a> |
        <a href="https://github.com/SHREEYANSHGIT" target="_blank">
            💻 GitHub
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

