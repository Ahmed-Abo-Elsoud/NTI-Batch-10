import streamlit as st
import pandas as pd
import joblib
import numpy as np

@st.cache_resource
def load_model():
    return joblib.load('diabetes_prediction_model.pkl')

model = load_model()

st.title("🩺 Diabetes Prediction App")
st.write("Enter the patient's details below to predict the likelihood of diabetes.")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    age = st.number_input("Age", min_value=0.0, max_value=120.0, value=30.0, step=1.0)
    hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    smoking_history = st.selectbox("Smoking History", ["never", "No Info", "current", "former", "ever", "not current"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0, format="%.2f")
    hba1c_level = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5, step=0.1)
    blood_glucose_level = st.number_input("Blood Glucose Level", min_value=50.0, max_value=400.0, value=100.0, step=1.0)

if st.button("Predict Diabetes Risk"):
    input_dict = {
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'bmi': bmi,
        'HbA1c_level': hba1c_level,
        'blood_glucose_level': blood_glucose_level,
        
        'gender_Female': 1 if gender == "Female" else 0,
        'gender_Male': 1 if gender == "Male" else 0,
        'gender_Other': 1 if gender == "Other" else 0,
        
        'smoking_history_No Info': 1 if smoking_history == "No Info" else 0,
        'smoking_history_current': 1 if smoking_history == "current" else 0,
        'smoking_history_ever': 1 if smoking_history == "ever" else 0,
        'smoking_history_former': 1 if smoking_history == "former" else 0,
        'smoking_history_never': 1 if smoking_history == "never" else 0,
        'smoking_history_not current': 1 if smoking_history == "not current" else 0,
    }

    input_data = pd.DataFrame([input_dict])

    try:
        if hasattr(model, "feature_names_in_"):
            input_data = input_data[model.feature_names_in_]

        prediction = model.predict(input_data)
        
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_data)[0][1] * 100
        else:
            probability = None

        st.write("---")
        st.subheader("Prediction Result:")
        
        if prediction[0] == 1:
            st.error("⚠️ The model indicates a **High Risk** of diabetes.")
        else:
            st.success("✅ The model indicates a **Low Risk** of diabetes.")
            
        if probability is not None:
            st.info(f"Estimated Probability: **{probability:.2f}%**")
            st.progress(int(probability) / 100)

        st.write("---")
        st.subheader("📊 Model Insights")
        
        importances = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "estimators_"):
            try:
                importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
            except Exception:
                pass
        
        if importances is not None and hasattr(model, "feature_names_in_"):
            feat_imp_df = pd.DataFrame({
                'Feature': model.feature_names_in_,
                'Importance': importances
            }).sort_values(by='Importance', ascending=True)
            
            st.write("**Feature Importance:**")
            st.write("This chart shows how much weight the model gives to each input factor when making its decision.")
            
            st.bar_chart(data=feat_imp_df.set_index('Feature'))
        else:
            st.write("**Patient Vitals vs. High Risk Baselines:**")
            comparison_df = pd.DataFrame({
                'Patient Value': [hba1c_level, blood_glucose_level, bmi],
                'High Risk Threshold (Approx)': [6.5, 126.0, 30.0]
            }, index=['HbA1c Level', 'Blood Glucose Level', 'BMI'])
            
            st.bar_chart(comparison_df)
            
    except Exception as e:
        st.error(f"Error making prediction: {e}")