import streamlit as st
import os
import sys
import pandas as pd

# Add the src directory to the path so we can import predict
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, 'src')
sys.path.append(src_dir)

from predict import load_model, predict_placement

st.set_page_config(page_title="Job Placement Prediction", layout="centered")

st.title("Job Placement Prediction using Naive Bayes")
st.markdown("Prediction generated using a Gaussian Naive Bayes model trained on student placement data.")
st.markdown("---")

model_path = os.path.join(base_dir, 'models', 'placement_naive_bayes.pkl')

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

with st.form("prediction_form"):
    st.subheader("Student Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", options=["M", "F"])
        ssc_p = st.number_input("SSC Percentage (10th)", min_value=0.0, max_value=100.0, value=60.0)
        ssc_b = st.selectbox("SSC Board", options=["Central", "Others"])
        hsc_p = st.number_input("HSC Percentage (12th)", min_value=0.0, max_value=100.0, value=60.0)
        hsc_b = st.selectbox("HSC Board", options=["Central", "Others"])
        hsc_s = st.selectbox("HSC Stream", options=["Commerce", "Science", "Arts"])
        
    with col2:
        degree_p = st.number_input("Degree Percentage", min_value=0.0, max_value=100.0, value=60.0)
        degree_t = st.selectbox("Degree Type", options=["Sci&Tech", "Comm&Mgmt", "Others"])
        workex = st.selectbox("Work Experience", options=["Yes", "No"])
        etest_p = st.number_input("Employability Test Percentage", min_value=0.0, max_value=100.0, value=60.0)
        specialisation = st.selectbox("MBA Specialisation", options=["Mkt&HR", "Mkt&Fin"])
        mba_p = st.number_input("MBA Percentage", min_value=0.0, max_value=100.0, value=60.0)
        
    submit_button = st.form_submit_button("Predict Placement")
    
if submit_button:
    student_data = {
        'gender': gender,
        'ssc_p': ssc_p,
        'ssc_b': ssc_b,
        'hsc_p': hsc_p,
        'hsc_b': hsc_b,
        'hsc_s': hsc_s,
        'degree_p': degree_p,
        'degree_t': degree_t,
        'workex': workex,
        'etest_p': etest_p,
        'specialisation': specialisation,
        'mba_p': mba_p
    }
    
    try:
        status, p_placed, p_not_placed = predict_placement(model, student_data)
        
        st.markdown("---")
        st.subheader("Prediction Result")
        
        if status == "Placed":
            st.success(f"**{status}**")
        else:
            st.error(f"**{status}**")
            
        st.write("### Probability")
        st.info(f"Placement probability: **{p_placed:.2f}%**")
        st.write(f"Not placed probability: {p_not_placed:.2f}%")
        
        st.markdown("*Note: This is a predictive model and does not guarantee that a student will get a job.*")
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
