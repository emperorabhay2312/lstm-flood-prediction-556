import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

st.title("Flood Prediction using LSTM")

# Load dataset
data = pd.read_csv("dataset/flood_train.csv")

st.subheader("Dataset Preview")
st.write(data.head())

# Assume last column is target
dataset = data.values
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# Create sequences
def create_dataset(data, time_step=5):
    X, Y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step), 0])
        Y.append(data[i+time_step, 0])
    return np.array(X), np.array(Y)

time_step = 5
X, y = create_dataset(scaled_data, time_step)

X = X.reshape(X.shape[0], X.shape[1], 1)

# Build model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step,1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

# Train button
if st.button("Train Model"):
    model.fit(X, y, epochs=5, batch_size=32, verbose=1)
    st.success("Model Trained Successfully!")

    # Prediction
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)

    # Plot
    st.subheader("Prediction Graph")
    fig, ax = plt.subplots()
    ax.plot(predictions, label="Predicted")
    ax.legend()
    st.pyplot(fig)