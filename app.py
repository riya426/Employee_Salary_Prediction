import streamlit as st
import numpy as np
import joblib

# Load model and encoders
model = joblib.load('dataset.pkl')
gender_encoder = joblib.load('models/gender_encoder.pkl')
education_encoder = joblib.load('models/education_encoder.pkl')
job_encoder = joblib.load('models/job_encoder.pkl')

# Streamlit UI
st.set_page_config(page_title="Employee Salary Predictor", page_icon="💼")
st.title("💼 Employee Salary Prediction")

st.markdown("### Enter employee details:")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, step=1)
gender = st.selectbox("Gender", gender_encoder.classes_)
education = st.selectbox("Education Level", education_encoder.classes_)
job_title = st.selectbox("Job Title", job_encoder.classes_)
experience = st.number_input("Years of Experience", min_value=0, max_value=50, step=1)

# Prediction button
if st.button("🔍 Predict Salary (₹)"):
    try:
        # Encode categorical inputs
        gender_encoded = gender_encoder.transform([gender])[0]
        education_encoded = education_encoder.transform([education])[0]
        job_encoded = job_encoder.transform([job_title])[0]

        # Create feature array
        input_features = np.array([[age, gender_encoded, education_encoded, job_encoded, experience]])

        # Predict
        prediction = model.predict(input_features)[0]

        # Display result in Indian Rupees
        st.success(f"💰 Estimated Salary: ₹{prediction:,.2f}")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit and Machine Learning")
