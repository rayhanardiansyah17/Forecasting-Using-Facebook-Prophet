from prophet import Prophet
import pandas as pd

# Simulated time series data
dates = pd.date_range(start='2023-01-01', periods=365)
data = pd.DataFrame({'ds': dates, 'y': np.sin(np.linspace(0, 20, 365)) + np.random.normal(0, 0.3, 365)})

# Prophet model
model = Prophet()
model.fit(data)

# Future predictions
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Plot forecast
model.plot(forecast)
plt.title('Forecast Using Facebook Prophet')
plt.show()
