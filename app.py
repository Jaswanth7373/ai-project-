# Disease Prediction from Symptoms using Machine Learning and Streamlit

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset (link: https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset)
df = pd.read_csv("https://raw.githubusercontent.com/karan-rajpal/disease-prediction-dataset/main/dataset.csv")

# Preprocessing
symptom_cols = df.columns[:-1]  # all columns except the last one
disease_col = df.columns[-1]    # the last column is the disease

# Replace 'f' and 't' with 0 and 1 for binary symptom presence
df[symptom_cols] = df[symptom_cols].replace({'f': 0, 't': 1})

# Encode target labels
le = LabelEncoder()
df[disease_col] = le.fit_transform(df[disease_col])

# Split data
X = df[symptom_cols]
y = df[disease_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Streamlit UI
st.title("🩺 Disease Prediction from Symptoms")
st.write("Select the symptoms you are experiencing below:")

# User input
temp_input = {}
for col in symptom_cols:
    temp_input[col] = st.checkbox(col.replace('_', ' ').capitalize())

# Convert input to dataframe
user_input = pd.DataFrame([temp_input])

# Predict
if st.button("Predict Disease"):
    prediction = model.predict(user_input)[0]
    predicted_disease = le.inverse_transform([prediction])[0]
    st.success(f"✅ Predicted Disease: {predicted_disease}")

    # Optional: Add some basic advice
    st.info("⚠️ Please note: This prediction is AI-based and not a medical diagnosis. Always consult a doctor.")

# To run: Save this as app.py and run `streamlit run app.py` in your terminal
