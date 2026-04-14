import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Title
st.title("🌊 Flood Prediction using LSTM (Demo App)")

st.write("This app demonstrates flood prediction using time-series data.")

# Load dataset
try:
    data = pd.read_csv("dataset/flood_train.csv")
except:
    st.error("Dataset not found. Please check file path.")
    st.stop()

# Show dataset
st.subheader("📊 Dataset Preview")
st.write(data.head())

# 🔥 Clean data (IMPORTANT FIX)
data = data.select_dtypes(include=['number'])   # keep only numeric columns
data = data.dropna()                            # remove missing values

if data.empty:
    st.error("Dataset has no valid numeric data.")
    st.stop()

# Convert to numpy
dataset = data.values

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

st.success("✅ Data preprocessing completed")

# Show processed data
st.subheader("🔍 Processed Data Sample")
st.write(scaled_data[:5])

# Simple visualization
st.subheader("📈 Data Visualization")
fig, ax = plt.subplots()
ax.plot(scaled_data, label="Scaled Data")
ax.legend()
st.pyplot(fig)

# Demo prediction (no TensorFlow for cloud compatibility)
st.subheader("🤖 Prediction Demo")

if st.button("Run Prediction"):
    # Fake prediction (demo)
    prediction = np.mean(scaled_data[-5:])
    
    st.success(f"Predicted Flood Level (demo): {prediction:.4f}")

    st.info("⚠️ Note: This is a demo prediction. Full LSTM model runs locally.")

# Footer
st.write("---")
st.write("👨‍💻 Developed by Abhay")