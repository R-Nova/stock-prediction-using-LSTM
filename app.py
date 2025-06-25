import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.title("📊 Stock Price Prediction (Linear Regression)")

uploaded_file = st.file_uploader("Upload a stock CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    if 'Close' not in df.columns:
        st.warning("Your CSV must have a 'Close' column.")
    else:
        df['PrevClose'] = df['Close'].shift(1)
        df = df.dropna()
        X = df[['PrevClose']].values
        y = df['Close'].values

        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)

        st.subheader("📈 Actual vs Predicted")
        fig, ax = plt.subplots()
        ax.plot(df['Close'].values, label="Actual", alpha=0.7)
        ax.plot(predictions, label="Predicted", alpha=0.7)
        ax.legend()
        st.pyplot(fig)
