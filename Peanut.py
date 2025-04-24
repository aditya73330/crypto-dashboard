import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# Download last 5 years of BTC-USD data
data = yf.download('BTC-USD', start='2020-01-01', end='2024-12-31')
data = data[['Close']]
data.dropna(inplace=True)
data.head()
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Scale the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Function to create sequences for LSTM
def create_sequences(dataset, window_size=60):
    X, y = [], []
    for i in range(window_size, len(dataset)):
        X.append(dataset[i-window_size:i, 0])  # 60 previous prices
        y.append(dataset[i, 0])                # the next price
    return np.array(X), np.array(y)

# Create sequences
window_size = 60
X, y = create_sequences(scaled_data, window_size)

# Reshape input to (samples, time steps, features) for LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

# Print shapes
print("Input shape:", X.shape)
print("Target shape:", y.shape)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Train-test split (80% train, 20% test)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build the model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
# 1. Predict using the model
predicted_prices = model.predict(X_test)

# 2. Inverse transform to get actual price values
predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# 3. Get matching dates for the test data
test_dates = data.index[-len(real_prices):]

# 4. Plot with proper date labels
import matplotlib.pyplot as plt

plt.figure(figsize=(14,6))
plt.plot(test_dates, real_prices, color='black', label='Actual BTC Price')
plt.plot(test_dates, predicted_prices, color='green', label='Predicted BTC Price')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
# Save the trained LSTM model
model.save("crypto_price_predictor_lstm.h5")
print("Model saved successfully.")
import joblib

# Save the MinMaxScaler object
joblib.dump(scaler, "scaler.save")
print("Scaler saved successfully.")



