import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --- Set Custom Styles ---
st.markdown(
    """
    <style>
    body {
        background-color: white;
        font-family: 'Times New Roman', serif;
    }
    .streamlit-expanderHeader {
        font-family: 'Times New Roman', serif;
    }
    .streamlit-container {
        font-family: 'Times New Roman', serif;
    }
    </style>
    """, unsafe_allow_html=True
)

# --- Healthy Weight Prediction Function ---
def healthy_weight_prediction(height, current_weight):
    # Convert height to meters
    height_m = height / 100  # Assuming height is in cm

    # Set the target BMI range
    min_bmi = 18.5
    max_bmi = 24.9

    # Calculate healthy weight range using BMI formula: weight = BMI * height^2
    min_healthy_weight = min_bmi * (height_m ** 2)
    max_healthy_weight = max_bmi * (height_m ** 2)

    return round(min_healthy_weight, 2), round(max_healthy_weight, 2)

# --- Class Scheduling Optimization ---
# Sample data for class attendance (for optimization demo purposes)
class_data = {
    "class_time": ["6 AM", "8 AM", "10 AM", "12 PM", "2 PM", "4 PM", "6 PM", "8 PM"],
    "num_attendees": [10, 20, 15, 12, 30, 25, 40, 35]
}

# Create a DataFrame to analyze class attendance
attendance_df = pd.DataFrame(class_data)

# Encode class time into numerical values for regression model (just a simple approach)
time_mapping = {
    "6 AM": 6, "8 AM": 8, "10 AM": 10, "12 PM": 12, "2 PM": 14, "4 PM": 16, "6 PM": 18, "8 PM": 20
}
attendance_df["time_num"] = attendance_df["class_time"].map(time_mapping)

# Train a simple linear regression model to predict attendees based on class time
X = attendance_df[["time_num"]]
y = attendance_df["num_attendees"]

model = LinearRegression()
model.fit(X, y)

# --- User Authentication ---
# Simulated user database (for demo purposes)
user_db = {
    'testuser': {'password': 'password123'},
    'newuser': {'password': 'newpassword'}
}

# Streamlit authentication process
def login():
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')

    if st.button("Login"):
        if username in user_db and user_db[username]['password'] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome {username}!")
        else:
            st.error("Invalid credentials. Please try again.")

def signup():
    st.subheader("Signup")
    username = st.text_input("Create Username")
    password = st.text_input("Create Password", type='password')
    
    if st.button("Signup"):
        if username in user_db:
            st.error("Username already exists. Please choose another.")
        else:
            user_db[username] = {'password': password}
            st.success(f"Account created successfully! Please login with your new username and password.")

# --- Main Application ---
def main():
    st.title("Fitness Studio: Class Scheduling Optimization & Healthy Weight Prediction")

    # --- Weight Prediction Section ---
    st.header("Healthy Weight Prediction")

    # Input height and current weight
    height = st.number_input("Enter your height (in cm):", min_value=50, max_value=250, value=170)
    current_weight = st.number_input("Enter your current weight (in kg):", min_value=10, max_value=200, value=70)

    if height and current_weight:
        min_weight, max_weight = healthy_weight_prediction(height, current_weight)
        st.write(f"Your healthy weight range is between **{min_weight} kg** and **{max_weight} kg** based on your height of **{height} cm**.")

    # --- Class Scheduling Section ---
    st.header("Class Attendance Optimization")

    # Display the class attendance data as a table
    st.write("### Class Attendance Data")
    st.dataframe(attendance_df)

    # Visualizing the class attendance
    st.write("### Class Attendance Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(attendance_df["class_time"], attendance_df["num_attendees"], color="skyblue")
    ax.set_xlabel("Class Time")
    ax.set_ylabel("Number of Attendees")
    ax.set_title("Class Attendance Analysis")
    st.pyplot(fig)

    # --- Predicting Attendance for a New Time ---
    st.write("### Predict Attendance for a New Class Time")

    # Input a class time for prediction
    class_time_input = st.selectbox("Select a class time:", attendance_df["class_time"])

    # Predict the attendance for the selected class time
    class_time_num = time_mapping[class_time_input]
    predicted_attendance = model.predict([[class_time_num]])

    st.write(f"Predicted attendance at **{class_time_input}** is: **{predicted_attendance[0]:.0f}** attendees")

    # --- Visualizing Predicted vs Actual Attendance ---
    st.write("### Predicted vs Actual Class Attendance")
    predicted_attendance_values = model.predict(X)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(attendance_df["class_time"], predicted_attendance_values, marker="o", linestyle="--", color="orange", label="Predicted Attendance")
    ax2.bar(attendance_df["class_time"], attendance_df["num_attendees"], color="skyblue", alpha=0.5, label="Actual Attendance")
    ax2.set_xlabel("Class Time")
    ax2.set_ylabel("Number of Attendees")
    ax2.set_title("Class Attendance Optimization - Actual vs Predicted")
    ax2.legend()
    st.pyplot(fig2)

# --- Streamlit Application Logic ---
if 'logged_in' not in st.session_state or not st.session_state.logged_in:
    # User is not logged in, show login/signup form
    action = st.radio("Choose an action", ['Login', 'Signup'])
    if action == 'Login':
        login()
    elif action == 'Signup':
        signup()
else:
    # User is logged in, show the main app
    main()
