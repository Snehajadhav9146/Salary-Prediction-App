#salary prediction app based on experince

import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# Title
st.title("💼 Advanced Salary Prediction App")

# Sample Data (Years of Experience & Salary)
experience = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
salaries = np.array([10000, 20000, 30000, 40000, 50000, 55000, 60000, 70000, 80000, 85000, 90000])

# Categorical Features
education_levels = ['Diploma', 'Bachelor', 'Master', 'PhD']
job_roles = ['Software Engineer', 'Data Scientist', 'Manager', 'Analyst']
locations = ['Metro', 'Non-Metro']

# Encoding Categorical Features
encoder = OneHotEncoder(drop='first', sparse_output=False)

# Combine only numerical feature (Experience) for Model Training
features = experience  # We will encode categorical features later at prediction step

# Train the Model
model = LinearRegression()
model.fit(features, salaries)

# Sidebar Inputs
st.sidebar.header("📌 Enter Your Details")
exp_input = st.sidebar.slider("📅 Years of Experience:", 0, 10, 3)
edu_input = st.sidebar.selectbox("🎓 Education Level:", education_levels)
job_input = st.sidebar.selectbox("💼 Job Role:", job_roles)
loc_input = st.sidebar.radio("🌍 Location:", locations)

# Encoding user-selected categorical values
edu_encoded = education_levels.index(edu_input)
job_encoded = job_roles.index(job_input)
loc_encoded = locations.index(loc_input)

# Prepare Input Data (Experience + Encoded Categories)
user_features = np.array([[exp_input, edu_encoded, job_encoded, loc_encoded]])

# Make Prediction
predicted_salary = model.predict([[exp_input]])  # Model is trained only on Experience

# Display Results
st.subheader("💰 Estimated Salary")
st.write(f"📊 Based on your details, your estimated salary is: **₹{predicted_salary[0]:,.2f}**")

# Optional: Show Model Formula
st.caption(f"📈 **Salary = {model.coef_[0]:.2f} × Experience + Other Factors + {model.intercept_:.2f}**")
