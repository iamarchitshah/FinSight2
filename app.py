# app.py 
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import matplotlib.pyplot as plt

from utils import fetch_stock_data, preprocess_data, create_dataset, ALPHA_VANTAGE_API_KEY
from models import train_lstm_model, train_rnn_model, train_svm_model, train_random_forest_model, make_predictions, evaluate_model

def main():
    st.title("Stock Price Prediction App")

    st.sidebar.header("User Input")
    ticker_symbol = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")

    # Date range for historical data
    today = date.today()
    start_date_default = today - timedelta(days=365 * 5) # 5 years of data
    end_date_default = today - timedelta(days=30) # End 30 days ago to avoid recent incomplete data

    start_date = st.sidebar.date_input("Start Date", start_date_default)
    end_date = st.sidebar.date_input("End Date", end_date_default)

    model_choice = st.sidebar.selectbox(
        "Select Prediction Model",
        ("LSTM", "RNN", "SVM", "Random Forest")
    )

    future_days = st.sidebar.slider("Days to Predict into Future", 1, 30, 7)

    if st.sidebar.button("Predict"):    
        if ALPHA_VANTAGE_API_KEY == "YOUR_ALPHA_VANTAGE_API_KEY":
            st.error("Please replace 'YOUR_ALPHA_VANTAGE_API_KEY' in utils.py with your actual Alpha Vantage API key.")
            return

        st.subheader(f"Predicting {ticker_symbol} Stock Prices")
        
        # Fetch data
        with st.spinner("Fetching historical data..."):
            stock_df = fetch_stock_data(ticker_symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        if stock_df is not None and not stock_df.empty:
            st.success("Data fetched successfully!")
            st.write(f"Historical data from {start_date} to {end_date}")
            st.write(stock_df.head())
            
            # Preprocess data
            with st.spinner("Preprocessing data..."):
                time_step = 100 # Can be adjusted or made user input
                X, y, scaler, scaled_data = preprocess_data(stock_df, time_step)
                
                # Splitting into training and testing data
                training_size = int(len(X) * 0.80)
                X_train, X_test = X[0:training_size], X[training_size:len(X)]
                y_train, y_test = y[0:training_size], y[training_size:len(y)]
                
                # Reshape for LSTM/RNN
                if model_choice in ["LSTM", "RNN"]:
                    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            
            st.success("Data preprocessed successfully!")
            
            if len(X_train) == 0 or len(X_test) == 0:
                st.warning("Not enough historical data available for the selected date range and time step to train and test the model. Please adjust the date range or reduce the time step.")
                return

            # Train and predict based on model choice
            model = None
            with st.spinner(f"Training {model_choice} model..."):
                if model_choice == "LSTM":
                    model = train_lstm_model(X_train, y_train, time_step)
                elif model_choice == "RNN":
                    model = train_rnn_model(X_train, y_train, time_step)
                elif model_choice == "SVM":
                    model = train_svm_model(X_train, y_train)
                elif model_choice == "Random Forest":
                    model = train_random_forest_model(X_train, y_train)
            
            if model:
                st.success(f"{model_choice} model trained successfully!")
                
                # Make predictions
                test_predict = make_predictions(model, X_test, scaler)
                
                # Evaluate model
                rmse = evaluate_model(scaler.inverse_transform(y_test.reshape(-1, 1)), test_predict)
                st.write(f"Model Performance (RMSE): {rmse:.4f}")
                
                # Prepare for future predictions
                last_data = scaled_data[len(scaled_data) - time_step:].reshape(1, -1)
                temp_input = list(last_data[0])
                
                # Predict future N days
                future_predictions = []
                i = 0
                while(i < future_days):
                    if model_choice in ["LSTM", "RNN"]:
                        if len(temp_input) > time_step:
                            last_input = np.array(temp_input[1:]).reshape(1, -1)
                            last_input = last_input.reshape((1, time_step, 1))
                            yhat = model.predict(last_input, verbose=0)[0]
                            temp_input.extend(yhat.tolist())
                            temp_input = temp_input[1:]
                            future_predictions.append(yhat)
                        else:
                            last_input = np.array(temp_input).reshape(1, -1)
                            last_input = last_input.reshape((1, time_step, 1))
                            yhat = model.predict(last_input, verbose=0)[0]
                            temp_input.extend(yhat.tolist())
                            future_predictions.append(yhat)
                    else:
                        # For SVM and Random Forest
                        last_input = np.array(temp_input).reshape(1, -1)
                        yhat = model.predict(last_input)[0]
                        temp_input.extend([yhat])
                        temp_input = temp_input[1:] # Ensure temp_input maintains fixed size
                        future_predictions.append([yhat])
                    i = i + 1
                
                future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
                
                # Create dates for future predictions
                last_date = stock_df.index[-1]
                future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=future_days)
                future_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Predicted Close'])
                
                st.subheader("Predicted Prices for Next N Days")
                st.write(future_df)
                
                # Plotting
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(stock_df.index, stock_df['Close'], label='Historical Prices')
                
                # Plot test predictions
                train_data_len = len(stock_df) - len(y_test)
                test_dates = stock_df.index[train_data_len:len(stock_df)]
                ax.plot(test_dates, test_predict, label='Test Predictions')
                
                # Plot future predictions
                ax.plot(future_df.index, future_df['Predicted Close'], label=f'Predicted Prices ({future_days} days)')
                
                ax.set_title(f'{ticker_symbol} Stock Price Prediction')
                ax.set_xlabel('Date')
                ax.set_ylabel('Price')
                ax.legend()
                st.pyplot(fig)

if __name__ == "__main__":
    main()
