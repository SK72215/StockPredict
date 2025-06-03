import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import yfinance as yf
from fuzzywuzzy import process
import requests
import json
import os
import ssl
import certifi
import urllib.request
from io import StringIO
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="AI Stock Predict", layout="wide")

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

def get_stocktwits_trending():
    try:
        response = requests.get("https://api.stocktwits.com/api/2/trending/symbols.json")
        if response.status_code == 200:
            data = response.json()
            trending = [symbol_data["symbol"] for symbol_data in data["symbols"]]
            return trending[:10]  # Top 10
        else:
            return []
    except Exception as e:
        print(f"Stocktwits fetch error: {e}")
        return []

@st.cache_data(show_spinner=False)
def fetch_trending_stocks():
    reddit_sentiment = []
    twitter_buzz = []
    stock_wits = get_stocktwits_trending()

    all_sources = reddit_sentiment + twitter_buzz + stock_wits
    weighted_counts = pd.Series(all_sources).value_counts()
    top_trending = weighted_counts[weighted_counts > 1].index.tolist()
    return top_trending

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    return []

def save_portfolio(portfolio):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f)

name_to_symbol, top_100, df, columns = load_company_data()
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


if st.sidebar.button("✨ Add Trending Stocks"):
    trending = fetch_trending_stocks()
    st.sidebar.info(f"Trending: {', '.join(trending)}")
    new_portfolio = list(set(portfolio + trending))
    save_portfolio(new_portfolio)
    st.rerun()

forecast_days = st.sidebar.selectbox("Select Forecast Period:", options=[10, 30], index=1)
run_analysis = st.sidebar.button("📊 Run Portfolio Analysis")

@st.cache_data(show_spinner=True)
def fetch_data(sym):
    return yf.download(sym, period="6mo")

if not portfolio or not run_analysis:
    st.warning("Select a portfolio and click 'Run Portfolio Analysis' to begin.")
    st.stop()

metrics_records = []

for symbol in portfolio:
    data = fetch_data(symbol)
    if data.empty:
        st.error(f"No data available for {symbol}. Please check the symbol or try another.")
        continue

    prophet_df = data.reset_index()[["Date", "Close"]].copy()
    prophet_df.columns = ["ds", "y"]
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    prophet_df['y'] = pd.to_numeric(prophet_df['y'], errors='coerce')
    prophet_df = prophet_df.dropna(subset=['ds', 'y'])

    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    trend_direction = "Up" if forecast['yhat'].iloc[-1] > forecast['yhat'].iloc[-forecast_days-1] else "Down"
    forecast_in_sample = forecast.iloc[:len(prophet_df)]
    mae = mean_absolute_error(prophet_df['y'], forecast_in_sample['yhat'])
    rmse = np.sqrt(mean_squared_error(prophet_df['y'], forecast_in_sample['yhat']))
    r2 = r2_score(prophet_df['y'], forecast_in_sample['yhat'])

    with st.expander(f"📈 {symbol} Analysis", expanded=False):
        col1, col2 = st.columns(2)
        col1.metric("Predicted Trend", trend_direction)
        col2.metric("Confidence", f"{r2 * 100:.2f}%")

        forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        forecast_df['type'] = ['Forecast' if i >= len(prophet_df) else 'Historical' for i in range(len(forecast_df))]

        chart_title = f"Historical and {forecast_days}-Day Forecast Chart"
        st.markdown(f"#### 📈 {chart_title}")

        combined_chart = alt.Chart(forecast_df).mark_line().encode(
            x=alt.X('ds:T', title='Date'),
            y=alt.Y('yhat:Q', title='Price'),
            color=alt.Color('type:N', title='Type'),
            tooltip=[
                alt.Tooltip('ds:T', title='Date'),
                alt.Tooltip('yhat:Q', title='Predicted Price'),
                alt.Tooltip('yhat_lower:Q', title='Lower Bound'),
                alt.Tooltip('yhat_upper:Q', title='Upper Bound')
            ]
        )

        band = alt.Chart(forecast_df).transform_filter(alt.datum.type == 'Forecast').mark_area(opacity=0.3).encode(
            x='ds:T',
            y='yhat_lower:Q',
            y2='yhat_upper:Q'
        )

        st.altair_chart(band + combined_chart, use_container_width=True)

        merged = pd.merge(prophet_df, forecast[['ds', 'yhat']], how='left', on='ds')
        merged['Error'] = merged['y'] - merged['yhat']

        st.markdown("#### 🔍 Forecast vs Actual Price Error")
        error_display = merged[['ds', 'y', 'yhat', 'Error']].tail(forecast_days)
        error_display.columns = ['Date', 'Actual Price', 'Predicted Price', 'Error']
        error_display['Date'] = pd.to_datetime(error_display['Date']).dt.strftime('%Y-%m-%d')
        st.dataframe(error_display.style.format({
            "Actual Price": "{:.2f}",
            "Predicted Price": "{:.2f}",
            "Error": "{:+.2f}"
        }).background_gradient(subset=["Error"], cmap="coolwarm"), use_container_width=True)

        metrics_records.append({
            "Symbol": symbol,
            "MAE": mae,
            "RMSE": rmse,
            "R²": r2 * 100
        })

if metrics_records:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔽 Portfolio Forecast Accuracy Summary")
        metrics_df = pd.DataFrame(metrics_records)
        styled_df = metrics_df.style.format({"R²": "{:.2f}%", "MAE": "{:.2f}", "RMSE": "{:.2f}"}) \
            .background_gradient(subset=["MAE"], cmap='Reds') \
            .background_gradient(subset=["RMSE"], cmap='Reds') \
            .background_gradient(subset=["R²"], cmap='Greens')
        st.dataframe(styled_df, use_container_width=True)

    with col2:
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
