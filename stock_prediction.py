import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Fetch historical data for Apple (AAPL)
ticker = 'AAPL'
data = yf.download(ticker, start='2020-01-01', end='2023-01-01')

# 2. Prepare features and target
data['Next_Close'] = data['Close'].shift(-1)
data = data.dropna()
features = data[['Open', 'High', 'Low', 'Volume']]
target = data['Next_Close']

# 3. Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)

# 4. Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict on test set
predictions = model.predict(X_test)

# 6. Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

# 7. Print metrics
print(f'Root Mean Squared Error (RMSE): ${rmse:.2f}')
print(f'R-squared (R²) Score: {r2:.4f}')

# 8. Plot actual vs predicted closing prices
plt.figure(figsize=(12,6))
plt.plot(y_test.values, label='Actual Close')
plt.plot(predictions, label='Predicted Close')
plt.title(f'Actual vs Predicted Closing Prices ({ticker})\nRMSE: ${rmse:.2f}, R²: {r2:.4f}')
plt.xlabel('Time (Test Set Index)')
plt.ylabel('Close Price ($)')
plt.legend()
plt.grid(True)
plt.show() 