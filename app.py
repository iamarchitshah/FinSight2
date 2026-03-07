import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import matplotlib.pyplot as plt

from utils import fetch_stock_data, preprocess_data, ALPHA_VANTAGE_API_KEY
from models import (
    train_lstm_model,
    train_rnn_model,
    train_svm_model,
    train_random_forest_model,
    make_predictions,
    evaluate_model
)

def main():

    st.title("📈 Stock Price Prediction App")

    # Sidebar
    st.sidebar.header("User Input")

    ticker_symbol = st.sidebar.text_input(
        "Enter Stock Ticker (Examples: AAPL, TSLA, MSFT)",
        "AAPL"
    ).strip().upper()

    today = date.today()

    start_date_default = today - timedelta(days=365)
    end_date_default = today

    start_date = st.sidebar.date_input("Start Date", start_date_default)
    end_date = st.sidebar.date_input("End Date", end_date_default)

    model_choice = st.sidebar.selectbox(
        "Select Prediction Model",
        ("LSTM", "RNN", "SVM", "Random Forest")
    )

    future_days = st.sidebar.slider("Days to Predict into Future", 1, 30, 7)

    if st.sidebar.button("Predict"):

        if ticker_symbol == "":
            st.error("Please enter a valid stock ticker.")
            return

        if ALPHA_VANTAGE_API_KEY == "YOUR_ALPHA_VANTAGE_API_KEY":
            st.error("Please add your Alpha Vantage API key in Streamlit secrets.")
            return

        st.subheader(f"Stock Prediction for {ticker_symbol}")

        # Fetch data
        with st.spinner("Fetching stock data..."):
            stock_df = fetch_stock_data(
                ticker_symbol,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )

        if stock_df is None or stock_df.empty:
            st.error("No stock data available. Please check the ticker.")
            return

        st.success("Data fetched successfully!")

        st.write("Historical Data")
        st.dataframe(stock_df.tail())

        st.write(f"Total Data Points: {len(stock_df)}")

        # Time step adjustment
        time_step = 60
        if len(stock_df) < time_step + 20:
            time_step = max(5, len(stock_df) // 3)
            st.info(f"Adjusted time_step to {time_step}")

        # Preprocessing
        with st.spinner("Preprocessing data..."):

            X, y, scaler, scaled_data = preprocess_data(stock_df, time_step)

            training_size = int(len(X) * 0.80)

            X_train = X[:training_size]
            X_test = X[training_size:]

            y_train = y[:training_size]
            y_test = y[training_size:]

            if model_choice in ["LSTM", "RNN"]:

                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        if len(X_train) == 0 or len(X_test) == 0:
            st.warning("Not enough data to train the model.")
            return

        # Train Model
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

        if model is None:
            st.error("Model training failed.")
            return

        st.success(f"{model_choice} model trained successfully!")

        # Predictions
        test_predict = make_predictions(model, X_test, scaler)

        rmse = evaluate_model(
            scaler.inverse_transform(y_test.reshape(-1, 1)),
            test_predict
        )

        st.write(f"Model RMSE: {rmse:.4f}")

        # Future prediction
        last_data = scaled_data[len(scaled_data) - time_step:].reshape(1, -1)

        temp_input = list(last_data[0])

        future_predictions = []

        for i in range(future_days):

            if model_choice in ["LSTM", "RNN"]:

                last_input = np.array(temp_input[-time_step:])
                last_input = last_input.reshape(1, time_step, 1)

                yhat = model.predict(last_input, verbose=0)[0]

                temp_input.append(yhat)
                future_predictions.append(yhat)

            else:

                last_input = np.array(temp_input[-time_step:]).reshape(1, -1)

                yhat = model.predict(last_input)[0]

                temp_input.append(yhat)
                future_predictions.append([yhat])

        future_predictions = scaler.inverse_transform(
            np.array(future_predictions).reshape(-1, 1)
        )

        # Create future dates
        last_date = stock_df.index[-1]

        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=future_days
        )

        future_df = pd.DataFrame(
            future_predictions,
            index=future_dates,
            columns=["Predicted Close"]
        )

        st.subheader("Future Predictions")
        st.dataframe(future_df)

        # Plot
        fig, ax = plt.subplots(figsize=(12,6))

        ax.plot(stock_df.index, stock_df["Close"], label="Historical")

        test_dates = stock_df.index[-len(test_predict):]

        ax.plot(test_dates, test_predict, label="Test Predictions")

        ax.plot(
            future_df.index,
            future_df["Predicted Close"],
            label="Future Predictions"
        )

        ax.set_title(f"{ticker_symbol} Stock Prediction")

        ax.set_xlabel("Date")
        ax.set_ylabel("Price")

        ax.legend()

        st.pyplot(fig)

if __name__ == "__main__":
    main()
