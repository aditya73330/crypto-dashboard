import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os
st.set_page_config(page_title="Crypto Forecast Dashboard", layout="wide")
st.markdown(
    """
    <style>
    .css-1d391kg {padding-top: 2rem;}
    .css-18e3th9 {padding-top: 2rem;}
    .main {background-color: #f5f5f5;}
    .block-container {padding: 2rem 2rem;}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- UI ----------------
st.title("📈 Multi-Crypto Price Prediction Dashboard")
crypto = st.selectbox("Choose a cryptocurrency:", ["BTC-USD", "ETH-USD", "DOGE-USD", "SOL-USD", "ADA-USD"])
start_date = st.date_input("Start Date", pd.to_datetime("2019-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2024-12-31"))
predict_btn = st.button("Predict")

# ---------------- Logic ----------------
if predict_btn:
    st.info("Fetching and processing data...")

    # Load data
    df = yf.download(crypto, start=start_date, end=end_date)
    data = df[["Close"]].dropna()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    sequence_length = 60
    X, y = [], []

    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Split into train/test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Load or train model
    model_path = f"{crypto.replace('-', '_')}_model.h5"
    scaler_path = f"{crypto.replace('-', '_')}_scaler.save"

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        st.success("Loaded existing model and scaler.")
    else:
        st.warning("Training new model...")
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32, verbose=0)

        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        st.success("Model trained and saved.")

    # Predict
    predicted = model.predict(X_test)
    predicted = scaler.inverse_transform(predicted)
    actual = scaler.inverse_transform(y_test)

    test_dates = data.index[-len(actual):]
    # --------- Future Forecast ---------
    st.subheader("🔮 Forecast Future Prices")

    future_days = st.slider("Select number of days to forecast", 1, 30, 7)
    future_input = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)

    future_predictions = []
    for _ in range(future_days):
        next_price = model.predict(future_input)[0]
        future_predictions.append(next_price)
        future_input = np.append(future_input[:, 1:, :], [[next_price]], axis=1)

    # Inverse scale
    future_predictions = scaler.inverse_transform(future_predictions)

    # Future date range
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

    # Plot future
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(future_dates, future_predictions, marker='o', color='orange', label="Forecasted")
    ax2.set_title(f"{crypto} - Forecasted Prices for Next {future_days} Days")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price (USD)")
    ax2.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    # Plot
    st.subheader(f"{crypto} - Actual vs Predicted Closing Price")
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(test_dates, actual, color='black', label='Actual')
    ax.plot(test_dates, predicted, color='green', label='Predicted')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.set_title(f"{crypto} Price Prediction")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)
