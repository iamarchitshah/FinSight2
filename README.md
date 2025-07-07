# 📊 FinSight: An Ensemble-Based Stock Market Prediction System using ML & DL

**FinSight2** is a powerful and user-friendly application that predicts the future performance of Indian stock market equities.  
It combines traditional Machine Learning models (**SVM, Random Forest**) with advanced Deep Learning techniques (**LSTM, RNN**) and crucial technical indicators (**RSI, Moving Averages, Bollinger Bands, MFI, MACD**) to generate accurate **7-day forecasts** for **Open, High, Low, and Close** prices.

---

## 🛠️ Features

* ✅ Predicts **Open, High, Low & Close** prices for the next **7 trading days**
* ✅ Supports all **Indian NSE-listed stocks** via Yahoo Finance (e.g. `RELIANCE.NS`)
* ✅ Uses **SVM, RF, LSTM, and RNN** models in an ensemble
* ✅ Enhances prediction with technical indicators: **RSI, EMA, SMA, WMA, MACD, Bollinger Bands, MFI**
* ✅ **Streamlit dashboard** with graphs, dark/light mode, and CSV export
* ✅ One-click export of predictions as **.csv**
* ✅ Toggle between **Light/Dark theme**

---

## 🔄 Architecture Overview

```
Historical Stock Data (Yahoo Finance)
       ⬇️
Add Technical Indicators (RSI, EMA, MACD, BB, MFI)
       ⬇️
Train ML/DL Models (SVM, RF, LSTM, RNN)
       ⬇️
Predict Next 7 Days (Open, High, Low, Close)
       ⬇️
Ensemble Averaging
       ⬇️
Display on Streamlit (Graphs + CSV Export)
```

---

## 📆 Dataset

* **Source**: Yahoo Finance (`yfinance` Python API)
* **Stocks Supported**: All NSE stocks (e.g., `INFY.NS`, `TCS.NS`, `HDFCBANK.NS`)
* **Timeframe**: User-selectable (e.g., Jan 2020 to Jul 2025)

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/iamarchitshah/FinSight2.git
cd FinSight2
pip install -r requirements.txt
streamlit run app.py
```

---

## 🚨 Requirements

```
streamlit
pandas
numpy
scikit-learn
yfinance
tensorflow
matplotlib
seaborn
ta
```

---

## 🌐 Deploy on Streamlit Cloud

1. Push your project to GitHub
2. Go to: [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **"New App"** and connect your repo
4. Set `app.py` as the entry point
5. Done! Share the public URL with others

---

## 👩‍💼 Authors

**Shah Archit**  
**Thakar Maitrey**  
Charotar University of Science and Technology  
Chandubhai S. Patel Institute of Technology  
Department: Information Technology  
Year: 3rd Year  
Semester: 5  
Batch: IT-2-D2

---

## 🏆 License

This project is licensed under the **MIT License**.
