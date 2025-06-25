import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.title("?? Stock Price Prediction with LSTM")

# Upload file
uploaded_file = st.file_uploader("Upload a stock CSV file (with a 'Close' column)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data Preview")
    st.write(df.head())

    if 'Close' not in df.columns:
        st.warning("Your CSV file must contain a 'Close' column.")
    else:
        close_prices = df['Close'].values.reshape(-1, 1)

        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        # Create sequences (last 60 days to predict next day)
        def create_sequences(data, seq_length=60):
            x, y = [], []
            for i in range(seq_length, len(data)):
                x.append(data[i-seq_length:i, 0])
                y.append(data[i, 0])
            return np.array(x), np.array(y)

        sequence_length = 60
        x, y = create_sequences(scaled_data, sequence_length)

        # Reshape input for LSTM
        x = x.reshape(x.shape[0], x.shape[1], 1)

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x.shape[1], 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        st.info("Training LSTM model... please wait (~20 sec)")
        model.fit(x, y, epochs=5, batch_size=32, verbose=0)

        # Make predictions
        predictions = model.predict(x)
        predictions = scaler.inverse_transform(predictions)
        actual = scaler.inverse_transform(y.reshape(-1, 1))

        # Plot results
        st.subheader("?? Actual vs Predicted Closing Prices")
        fig, ax = plt.subplots()
        ax.plot(actual, label="Actual", color='blue')
        ax.plot(predictions, label="Predicted", color='orange')
        ax.set_title("Stock Price Prediction")
        ax.set_xlabel("Days")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)