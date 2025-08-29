# FinSight2 📈
**Sentiment-Enhanced Deep Learning Stock Prediction**

FinSight2 is a deep learning–based system that predicts Indian stock market trends using historical price data and technical indicators. The focus is on **LSTM** and **RNN** models, with ensemble logic and an interactive **Streamlit dashboard** for visualization.  

---

## ✨ Features  
- Fetches real-time historical stock data using **Yahoo Finance (`yfinance`)**  
- Computes popular **technical indicators**:  
  - RSI (Relative Strength Index)  
  - EMA, SMA, WMA (Moving Averages)  
  - MACD (Moving Average Convergence Divergence)  
  - Bollinger Bands  
  - MFI (Money Flow Index)  
- Deep learning prediction with:  
  - **LSTM (Long Short-Term Memory)**  
  - **RNN (Recurrent Neural Network)**  
- Ensemble forecasting combining outputs from DL models  
- Interactive **Streamlit dashboard**:  
  - Input any NSE stock ticker  
  - Generate 7-day OHLC forecasts  
  - Toggle **light/dark mode**  
  - Export results as **CSV**  

---

## ⚙️ Installation  

Clone the repository:  
```bash
git clone https://github.com/iamarchitshah/FinSight2.git
cd FinSight2
```

Create a virtual environment (recommended):  
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

Install requirements:  
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage  

Run the Streamlit app:  
```bash
streamlit run app.py
```

Then open the local server link (e.g., `http://localhost:8501`) in your browser.  

---

## 📊 Example  

1. Enter stock ticker (e.g., `RELIANCE.NS`).  
2. Select analysis options (indicators, forecast period).  
3. View interactive plots:  
   - Historical vs Predicted Prices  
   - RSI, MACD, Bollinger Bands, etc.  
4. Export predictions to CSV.  

---

## 📂 Project Structure  

```
FinSight2/
│── app.py              # Streamlit dashboard
│── models.py           # LSTM and RNN model definitions
│── utils.py            # Data fetching & technical indicators
│── requirements.txt    # Dependencies
│── README.md           # Project documentation
```

---

## ✅ Roadmap  

- [ ] Add sentiment analysis from news & social media  
- [ ] Enhance ensemble strategy  
- [ ] Cloud deployment (Streamlit Cloud / AWS / GCP)  
- [ ] Add hyperparameter tuning module  

---

## 🗓️ 7-Day Intern Work Plan  

| Day | D24IT166 Focus                        | D24IT168 Focus                              |
|-----|----------------------------------------|-----------------------------------------------|
| 1   | Setup & repo exploration               | Environment setup & yfinance testing          |
| 2   | Data fetching framework                | Technical indicator implementation            |
| 3   | LSTM exploration & training            | RNN exploration & training                    |
| 4   | Ensemble logic (LSTM+RNN)              | Forecast generation (Open/High/Low/Close)     |
| 5   | Dashboard visual improvements          | CSV export & UI polish                        |
| 6   | Unit testing for core modules          | Edge case testing and error handling          |
| 7   | README & documentation updates         | Deployment prep & verification                |

---

## 🤝 Contributing  

Pull requests are welcome. For significant changes, open an issue to discuss what you’d like to modify.  

---

## 📜 License  

This project is licensed under the MIT License.  
