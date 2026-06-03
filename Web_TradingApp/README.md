Professional Trading Platform — Web Edition

Setup \& Run Guide

📁 File Structure

Place these files in one folder:

 Web_TradingApp
        ├── streamlit_app.py          ← Main web app (this file)
        ├── requirements_web.txt      ← Python dependencies
        ├── config.json                ← Your OKX API keys (keep secret!)
        ├── strategy_settings.json    ← Strategy parameters
        ├── models
        │   ├── __init__.py
        │   ├── base_ML.py
        │   ├── lstm_model_NEW.py
        │   ├── random_forest.py
        │   └── xgboost_model.py
        ├── strategies
        │   ├── __init__.py
        │   ├── base3_New.py
        │   ├── MomentumStrategy_MACD_HybridScore_Latest.py
        │   ├── KalmanTrendStrategy_New.py
        │   ├── TradingStrategy3.py
        │   ├── scalping_strategy.py
        │   └── monte_carlo_simulator.py
        └── utils
                ├── __init__.py 
                ├── AdaptiveWeightManager.py
                ├── FinancialCircletimer_New.py
                ├── utils.py
                └── WeightManager.py

⚡ Quick Start (Windows)
# 1. Open Command Prompt in your trading folder
C:\Users\dahab\PyCharm_2026.2.23\New_Bollinger_bands\Web_TradingApp
cd Web_TradingApp

# 2. Install dependencies
pip install -r requirements_web.txt

# 3. Run the app
streamlit run streamlit_app.py

The app will open automatically at: http://localhost:8501

🔑 Config Setup
Your config.json should look like:

{
         "demo": {
              "api\_key": "your\_demo\_key",
              "api\_secret\_key": "your\_demo\_secret",
              "passphrase": "your\_passphrase",
              "flag": "1"
         },
            "live": {
              "api\_key": "your\_live\_key",
              "api\_secret\_key": "your\_live\_secret",
              "passphrase": "your\_passphrase",
              "flag": "0"
            }
}

⚠️ NEVER share config.json or commit it to GitHub.

🚀 Features
Feature	Status
Dark trading UI	✅
Live/Demo/Backtest modes	✅
OKX API connection	✅
Candlestick charts (Plotly)	✅
EMA overlays on chart	✅
Real-time trading log	✅
Backtest engine	✅
Excel export	✅
ML predictions (RF/XGB/LSTM)	✅
Parameter management	✅
Risk management controls	✅
Emergency stop	✅
Help documentation	✅

Access from anywhere: http://your-server-ip:8501

⚠️ Notes

winsound and pyttsx3 (audio alerts) are not available in web mode
Tkinter GUI is fully replaced by Streamlit
All strategy logic, ML models, and backtesting remain unchanged
For live trading, ensure stable internet connection on the server

