import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import os

st.title('Sales Demand Forecasting')

# Sidebar for user input
horizon = st.sidebar.selectbox('Select forecast horizon (weeks):', [7, 14, 30], index=2)

# Data loading
uploaded_file = st.file_uploader("Upload your sales data CSV", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    # fallback to default data
    df = pd.read_csv('data/train.csv')

# Convert Date column
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
else:
    st.error('No Date column found in train.csv')
    st.stop()

# Optional: Store/Dept selection
stores = df['Store'].unique()
depts = df['Dept'].unique()
store = st.sidebar.selectbox('Select Store:', stores)
dept = st.sidebar.selectbox('Select Dept:', depts)

filtered = df[(df['Store'] == store) & (df['Dept'] == dept)]

# Aggregate sales by week (Date)
agg = filtered.groupby('Date')['Weekly_Sales'].sum().reset_index()
agg = agg.rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})

if len(agg) < horizon + 1:
    st.error('Not enough data for the selected store/dept and forecast horizon.')
    st.stop()

# Train/test split
train = agg.iloc[:-horizon]
test = agg.iloc[-horizon:]

# Fit Prophet model
m = Prophet()
m.fit(train)

# Forecast
future = m.make_future_dataframe(periods=horizon, freq='W-FRI')
forecast = m.predict(future)

# Plot forecast
fig1 = m.plot(forecast)
st.pyplot(fig1)
st.markdown("""
**What does this mean?**  
This plot shows the actual sales for the selected store and department (blue dots/line) and the predicted sales for the next few weeks (black line with shaded area). The shaded area represents the uncertainty in the forecast—the wider it is, the less certain the prediction.
""")

# Plot components
fig2 = m.plot_components(forecast)
st.pyplot(fig2)
st.markdown("""
**What does this mean?**  
These plots break down the forecast into different parts:
- **Trend:** The overall direction of sales over time (upward, downward, or flat).
- **Weekly/Yearly Seasonality:** Regular patterns that repeat every week or year (like higher sales on holidays).
- **Holidays:** The effect of special holidays on sales.
""")

# Evaluation metrics
# Merge forecast and test on 'ds' to avoid KeyError
merged = pd.merge(test, forecast[['ds', 'yhat']], on='ds', how='inner')
if len(merged) < len(test):
    st.warning('Some test dates are missing in the forecast. Evaluation is based on available dates only.')
if len(merged) == 0:
    st.error('No overlapping dates between test set and forecast. Cannot compute metrics.')
else:
    rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
    mae = mean_absolute_error(merged['y'], merged['yhat'])
    mape = np.mean(np.abs((merged['y'] - merged['yhat']) / merged['y'])) * 100

    st.subheader('Model Performance (last {} weeks)'.format(horizon))
    st.write(f'RMSE: {rmse:.2f}')
    st.write(f'MAE: {mae:.2f}')
    st.write(f'MAPE: {mape:.2f}%')
    st.markdown("""
**What does this mean?**  
These numbers show how accurate the model's predictions are for the last few weeks:
- **RMSE (Root Mean Squared Error):** On average, how much the predictions differ from the actual sales. Lower is better.
- **MAE (Mean Absolute Error):** The average absolute difference between predicted and actual sales. Lower is better.
- **MAPE (Mean Absolute Percentage Error):** The average error as a percentage of actual sales. For example, 10% means predictions are off by 10% on average.
""") 