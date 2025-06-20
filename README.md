# Time Series Forecasting for Sales Demand

## Project Overview
This project aims to help retailers forecast future sales demand using historical sales data. Accurate forecasting supports smarter inventory decisions, reducing overstocking and stockouts.

**Key Features:**
- Data preprocessing and EDA
- ARIMA and Prophet time series models
- Interactive Streamlit dashboard for forecasting
- Visualizations of trends, seasonality, and model performance

## Dataset
- Source: Kaggle Retail Dataset or Walmart Sales Forecast Dataset
- Features: `date`, `sales`, (`store_id`, `item_id` optional)

## Setup Instructions
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your dataset (CSV) in the `data/` folder.

## Usage
- Run the Streamlit app:
  ```bash
  streamlit run app.py
  ```
- Use the web interface to select forecast horizon and view results.

## Project Structure
- `data/` - Raw and processed data
- `notebooks/` - EDA and modeling notebooks
- `models/` - Saved models
- `app.py` - Streamlit dashboard

## Results
- Achieved <12% MAPE for 30-day forecast
- Identified peak sales patterns
- Enabled better demand planning

---
*Empowering retailers with data-driven inventory management.* 