#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import LSTM, Dense
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)


# In[28]:


df = pd.read_csv("D:\christ\Mentorness Machine Learning  Internship\MarketPricePrediction.csv")


# In[5]:


print(df.shape)


# In[6]:


# Display the first few rows of the dataframe
print(df.head())


# In[7]:


# Display information about the dataframe
print(df.info())


# In[8]:


# Summary statistics
print(df.describe())


# In[9]:


# Handling missing values
df.dropna(inplace=True)

# Encoding categorical variables
label_encoders = {}
for column in ["market", "state", "city"]:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Check if any missing value remains
print(df.isnull().sum())


# In[10]:


# Scaling numerical features
num_columns = ['year', 'priceMin', 'priceMax', 'priceMod']  # Assuming 'quantity' is the target column
scaler = MinMaxScaler()
X_train[num_columns] = scaler.fit_transform(X_train[num_columns])
X_test[num_columns] = scaler.transform(X_test[num_columns])


# In[11]:


# Step 2: Exploratory Data Analysis (EDA)
# Assuming "date" is already in datetime format
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
# Visualizing temporal patterns
plt.plot(df['quantity'])
plt.title('Quantity Over Time')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.show()


# In[8]:


# Decomposing time series to identify trends and seasonality
decomposition = seasonal_decompose(df['quantity'], model='additive', period=12)
decomposition.plot()
plt.show()


# In[9]:


# Check for stationarity using Augmented Dickey-Fuller test
result = adfuller(df['quantity'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])


# In[10]:


# Autocorrelation and Partial Autocorrelation plots
plot_acf(df['quantity'], lags=50)
plt.show()
plot_pacf(df['quantity'], lags=50)
plt.show()


# In[29]:


df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Define train and test splits
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# ARIMA forecast function
def arima_forecast(train, test):
    history = [x for x in train['priceMod']]
    predictions = []
    for t in range(len(test)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        history.append(test['priceMod'].iloc[t])
    return predictions

# Forecast using ARIMA
arima_predictions = arima_forecast(train, test)

# Evaluate forecasts
arima_mae = mean_absolute_error(test['priceMod'], arima_predictions)
arima_mse = mean_squared_error(test['priceMod'], arima_predictions)
arima_rmse = np.sqrt(arima_mse)

print("ARIMA MAE:", arima_mae)
print("ARIMA MSE:", arima_mse)
print("ARIMA RMSE:", arima_rmse)


# In[30]:


# Plot the forecasts against actual outcomes
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['priceMod'], label='Train')
plt.plot(test.index, test['priceMod'], label='Test', color='orange')
plt.plot(test.index, arima_predictions, label='Predicted', color='green')
plt.xlabel('Date')
plt.ylabel('PriceMod')
plt.title('ARIMA Model Forecast vs Actual')
plt.legend()
plt.show()


# In[6]:


# Prepare data for LSTM
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['quantity']])

# Create sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 12
X, y = create_sequences(scaled_data, seq_length)


# In[7]:


# Split data into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


# In[8]:


# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32)


# In[9]:


# Predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)


# In[10]:


# Plot predictions
plt.figure(figsize=(10, 5))
plt.plot(df.index[-len(y_test):], scaler.inverse_transform(y_test.reshape(-1, 1)), label='True Values')
plt.plot(df.index[-len(predictions):], predictions, label='Predicted Values')
plt.legend()
plt.show()


# In[11]:


# Calculate metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')


# In[13]:


pip install prophet


# In[20]:


df['date'] = pd.to_datetime(df['date'])
df = df[['date', 'priceMod']]  # Only keep relevant columns for Prophet
df.columns = ['ds', 'y']  # Rename columns to fit Prophet's requirements

# Define train and test splits
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]


# In[21]:


# Initialize and fit the Prophet model
prophet_model = Prophet()
prophet_model.fit(train)

# Make future dataframe and forecast
future = prophet_model.make_future_dataframe(periods=len(test))
forecast = prophet_model.predict(future)

# Evaluate forecasts
y_true = test['y'].values
y_pred = forecast['yhat'].iloc[-len(test):].values

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

print("Prophet MAE:", mae)
print("Prophet MSE:", mse)
print("Prophet RMSE:", rmse)


# In[22]:


# Plot the forecast
fig = prophet_model.plot(forecast)
fig.show()

