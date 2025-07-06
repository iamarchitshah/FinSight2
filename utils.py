# utils.py 
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import streamlit as st

# Get Alpha Vantage API key from Streamlit secrets
ALPHA_VANTAGE_API_KEY = st.secrets["ALPHA_VANTAGE_API_KEY"]

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetches historical stock data from Alpha Vantage.
    
    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        
    Returns:
        pd.DataFrame: DataFrame with historical stock data,
                      or None if an error occurs.
    """
    try:
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        # Using TIME_SERIES_DAILY which is available on the free tier
        data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
        
        # Select '4. close' which is the closing price for TIME_SERIES_DAILY
        df = data['4. close']
        df = df.iloc[::-1]  # Reverse to have oldest data first
        df.index = pd.to_datetime(df.index)
        
        # Filter by date range
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        return df.to_frame(name='Close')
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}. Please check your API key and internet connection, or if the ticker symbol is valid.")
        return None

def preprocess_data(df, time_step=1):
    """
    Preprocesses the stock data for model training.
    
    Args:
        df (pd.DataFrame): DataFrame with 'Close' prices.
        time_step (int): Number of previous time steps to use as features.
        
    Returns:
        tuple: Scaled data, MinMaxScaler object, X (features), y (labels).
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(len(scaled_data) - time_step):
        X.append(scaled_data[i:(i + time_step), 0])
        y.append(scaled_data[i + time_step, 0])
        
    return np.array(X), np.array(y), scaler, scaled_data

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY) 
