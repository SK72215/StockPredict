# app.py - Streamlit UI for deployment

import streamlit as st
from ai_stock_predictor.predictor import AIPredictor

st.set_page_config(page_title="AI Stock Predict", layout="wide")

st.markdown("""
<style>
    .big-font { font-size:32px !important; font-weight: 600; color: #2c3e50; }
    .subtle-text { font-size:14px; color: #7f8c8d; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-font'>📊 AI Stock Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='subtle-text'>Track. Analyze. Predict. Smarter.</div>", unsafe_allow_html=True)

symbols = st.text_input("🔍 Enter stock symbols", placeholder="e.g., AAPL, TSLA, NVDA")
interval = st.selectbox("⏱️ Prediction interval", ["daily", "hourly"])

st.divider()

tabs = st.tabs(["📈 Overview", "📉 Trends", "🧠 AI Predictions", "📰 News & Sentiment", "📊 Portfolio Tracker"])

with tabs[2]:  # AI Predictions Tab
    st.subheader("📊 AI-Based Predictions")
    if st.button("🔮 Predict Now"):
        predictor = AIPredictor()
        if symbols:
            cols = st.columns(len(symbols.split()))
            for idx, symbol in enumerate(symbols.upper().split()):
                result = predictor.predict(symbol, interval)
                with cols[idx]:
                    st.metric(
                        label=symbol,
                        value=result['prediction'],
                        delta=f"Confidence: {result['confidence']*100:.1f}%"
                    )
        else:
            st.warning("Please enter at least one stock symbol.")
