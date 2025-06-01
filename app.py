# app.py - Streamlit UI for deployment

import streamlit as st
from ai_stock_predictor.predictor import AIPredictor

st.title("📈 AI Stock Predictor")

symbols = st.text_input("Enter stock symbols (e.g., AAPL TSLA GOOGL):")
interval = st.selectbox("Select prediction interval:", ["daily", "hourly"])

if st.button("Predict"):
    predictor = AIPredictor()
    if symbols:
        st.subheader("Prediction Results")
        for symbol in symbols.upper().split():
            result = predictor.predict(symbol, interval)
            st.write(f"**{symbol}** → {result['prediction']} ({result['confidence']*100:.1f}% confidence)")
    else:
        st.warning("Please enter at least one stock symbol.")
