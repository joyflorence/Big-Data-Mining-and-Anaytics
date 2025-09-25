import streamlit as st
import joblib
import pandas as pd

model = joblib.load("pollution_model.pkl")

st.title("Pollution Prediction App")
st.write("Enter the feature values to predict pollution levels.")

feature_names = ['CRIM', 'AC', 'INDUS', 'LS', 'RM', 'AGE', 'DIS', 'RAD', 'PTRATIO', 'DMT', 'LSTAT', 'MO', 'TAX']

input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(f"{feature}", value=0.0)


input_df = pd.DataFrame([input_data])


if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Pollution Level: {prediction:.4f}")

if st.checkbox("Show Feature Importance"):
    importances = model.feature_importances_
    st.bar_chart(pd.Series(importances, index=feature_names))
