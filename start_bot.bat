@echo off
REM Double-click to start TradeAI. Opens on the Live Scalper with live TradingView data.
cd /d "%~dp0"
call trading_bot\Scripts\activate.bat
python -m streamlit run app.py
