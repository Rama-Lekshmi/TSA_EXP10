# Developed By: Rama E.K. Lekshmi
# Register Number : 212222240082

# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv('/mnt/data/AirPassengers.csv')
data['Month'] = pd.to_datetime(data['Month'], format='%Y-%m')
data.set_index('Month', inplace=True)
data.rename(columns={'#Passengers': 'passengers'}, inplace=True)

# Visualize the time series data
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['passengers'])
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.title('Air Passenger Time Series')
plt.show()

# Check for stationarity
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

check_stationarity(data['passengers'])

# ACF and PACF plots
plot_acf(data['passengers'])
plt.show()
plot_pacf(data['passengers'])
plt.show()

# Split the data into train and test sets (80% train, 20% test)
train_size = int(len(data) * 0.8)
train, test = data['passengers'][:train_size], data['passengers'][train_size:]

# SARIMA model configuration
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

# Forecasting
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Calculate RMSE
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.title('SARIMA Model Forecasting for Air Passengers')
plt.legend()
plt.show()

```
### OUTPUT:

![image](https://github.com/user-attachments/assets/8055a35d-3d02-4575-a257-90d047dd0905)

![image](https://github.com/user-attachments/assets/eb6dfce3-ffb2-4ca1-b436-60ba5bd5b1ce)

![image](https://github.com/user-attachments/assets/84f14efc-f1bb-42f6-8a33-501e586ce296)

![image](https://github.com/user-attachments/assets/76f09be8-17a0-49f1-8fa5-f5bfd042a628)


### RESULT:
Thus the program run successfully based on the SARIMA model.
