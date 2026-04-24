import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb

# Set page configuration
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

# Load the trained XGBoost model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# App Title
st.title("Student Performance Prediction App")
st.write("Enter the details below to predict the outcome.")

# Create input fields for the 9 required features
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender (0 or 1)", [0, 1])
    age = st.number_input("Age", min_value=10, max_value=100, value=15, step=1)
    study_hours_per_week = st.number_input("Study Hours per Week", min_value=0, max_value=100, value=10, step=1)
    parent_education = st.selectbox("Parent Education Level (Encoded Int)", [0, 1, 2, 3, 4])
    internet_access = st.selectbox("Internet Access (0=No, 1=Yes)", [0, 1])

with col2:
    # attendance_rate is the only float feature
    attendance_rate = st.number_input("Attendance Rate (0.0 to 1.0)", min_value=0.0, max_value=1.0, value=0.85, step=0.01)
    extracurricular = st.selectbox("Extracurricular Activities (0=No, 1=Yes)", [0, 1])
    previous_score = st.number_input("Previous Score", min_value=0, max_value=100, value=75, step=1)
    final_score = st.number_input("Final Score", min_value=0, max_value=100, value=80, step=1)

# Prediction Button
if st.button("Predict"):
    # Construct the exact dictionary structure expected by the model
    input_data = {
        'gender': [int(gender)],
        'age': [int(age)],
        'study_hours_per_week': [int(study_hours_per_week)],
        'attendance_rate': [float(attendance_rate)],
        'parent_education': [int(parent_education)],
        'internet_access': [int(internet_access)],
        'extracurricular': [int(extracurricular)],
        'previous_score': [int(previous_score)],
        'final_score': [int(final_score)]
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame(input_data)
    
    try:
        # Make Prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        # Display Results
        st.subheader("Results")
        if prediction == 1:
            st.success(f"Prediction: **Class 1** (Probability: {probability:.2%})")
        else:
            st.error(f"Prediction: **Class 0** (Probability: {(1-probability):.2%})")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
