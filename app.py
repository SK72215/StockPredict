import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import yfinance as yf
from ai_stock_predictor.predictor import AIPredictor
from fuzzywuzzy import process
import requests
import json
import os
import ssl
import certifi
import urllib.request
from io import StringIO
from prophet import Prophet

st.set_page_config(page_title="AI Stock Predict", layout="wide")
st.title("IntelliStock - Unlock the Potential")
st.caption("Smarter Investing Powered by AI – Research, Track, Predict")

PORTFOLIO_FILE = "portfolio.json"

@st.cache_data(show_spinner=False)
def load_company_data():
    url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"
    context = ssl.create_default_context(cafile=certifi.where())
    response = urllib.request.urlopen(url, context=context)
    content = response.read().decode('utf-8')
    df = pd.read_csv(StringIO(content), sep="|", names=["symbol", "name"])
    df.columns = [col.strip().lower() for col in df.columns]
    if 'name' not in df.columns or 'symbol' not in df.columns:
        st.error(f"CSV file is missing expected columns. Found: {df.columns.tolist()}")
        st.stop()
    name_to_symbol = {str(name).lower(): symbol for name, symbol in zip(df['name'], df['symbol'])}
    top_100 = df.head(100)[["symbol", "name"]].values.tolist()
    return name_to_symbol, top_100, df, df.columns.tolist()

name_to_symbol, top_100, df, columns = load_company_data()

# --- Portfolio Tracking ---
def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    return []

def save_portfolio(portfolio):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f)

portfolio = load_portfolio()

st.sidebar.header("Stock Portfolio Builder")
all_symbols = sorted(df['symbol'].dropna().unique().tolist())
selected_stocks = st.sidebar.multiselect("Select stocks to add to your portfolio:", all_symbols, default=portfolio if portfolio else None, key="portfolio_selector")

if selected_stocks != portfolio:
    save_portfolio(selected_stocks)
    portfolio = selected_stocks
    st.sidebar.success("Portfolio updated")

if st.sidebar.button("🗑️ Clear Portfolio"):
    portfolio = []
    save_portfolio(portfolio)
    st.sidebar.warning("Portfolio cleared")

# --- Load Data and Predictions ---
@st.cache_data(show_spinner=True)
def fetch_data(sym):
    return yf.download(sym, period="6mo")

if portfolio:
    symbol = portfolio[0]  # Default to first symbol for predictions
else:
    st.warning("Portfolio is empty. Please select stocks from the sidebar.")
    st.stop()

data = fetch_data(symbol)

if data.empty:
    st.error(f"No data available for {symbol}. Please check the symbol or try another.")
    st.stop()

predictor = AIPredictor()
prediction = predictor.predict(symbol, data)

# --- Layout ---
st.subheader(f"📊 AI Prediction for {symbol}")
st.metric("Predicted Trend", prediction.get("trend", "N/A"))
st.metric("Confidence", f"{prediction.get('confidence', 0)*100:.1f}%")

# --- Prophet Model Prediction ---
st.subheader("🔮 10-Day Forecast with Confidence Interval (Prophet)")

prophet_df = data.reset_index()[["Date", "Close"]].copy()
prophet_df.columns = ["ds", "y"]
prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
prophet_df['y'] = pd.to_numeric(prophet_df['y'], errors='coerce')
prophet_df = prophet_df.dropna(subset=['ds', 'y'])

model = Prophet()
model.fit(prophet_df)

future = model.make_future_dataframe(periods=10)
forecast = model.predict(future)

forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)

forecast_chart = alt.Chart(forecast_df).mark_line(color='blue').encode(
    x=alt.X('ds:T', title='Date'),
    y=alt.Y('yhat:Q', title='Predicted Price'),
    tooltip=[
        alt.Tooltip('ds:T', title='Date'),
        alt.Tooltip('yhat:Q', title='Forecasted Price')
    ]
).interactive()

band = alt.Chart(forecast_df).mark_area(opacity=0.3).encode(
    x='ds:T',
    y='yhat_lower:Q',
    y2='yhat_upper:Q'
)

st.altair_chart(band + forecast_chart, use_container_width=True)

# --- Portfolio Performance ---
if portfolio:
    st.subheader("📈 Portfolio Historical Prices")
    all_hist_data = []
    for stock in portfolio:
        df_stock = fetch_data(stock)
        if not df_stock.empty:
            df_stock = df_stock[['Close']].copy()
            df_stock['Symbol'] = stock
            df_stock['Date'] = df_stock.index
            all_hist_data.append(df_stock)
    if all_hist_data:
        merged_df = pd.concat(all_hist_data)
        chart = alt.Chart(merged_df).mark_line().encode(
            x=alt.X('Date:T', title='Date'),
            y=alt.Y('Close:Q', title='Price'),
            color=alt.Color('Symbol:N', title='Stock Symbol'),
            tooltip=['Date:T', 'Symbol:N', 'Close:Q']
        ).interactive()
        st.altair_chart(chart, use_container_width=True)

    st.subheader("📈 Portfolio Normalized Performance")
    normalized_df = pd.DataFrame()
    for stock in portfolio:
        df_stock = fetch_data(stock)
        if not df_stock.empty:
            df_temp = df_stock[['Close']].copy()
            df_temp['Symbol'] = stock
            df_temp['Normalized'] = df_temp['Close'] / df_temp['Close'].iloc[0]
            df_temp['Date'] = df_temp.index
            normalized_df = pd.concat([normalized_df, df_temp])
    if not normalized_df.empty:
        norm_chart = alt.Chart(normalized_df).mark_line().encode(
            x=alt.X('Date:T', title='Date'),
            y=alt.Y('Normalized:Q', title='Normalized Price'),
            color=alt.Color('Symbol:N', title='Stock Symbol'),
            tooltip=['Date:T', 'Symbol:N', 'Normalized:Q']
        ).interactive()
        st.altair_chart(norm_chart, use_container_width=True)
else:
    st.info("Your portfolio is empty. Add stocks using the sidebar.")
