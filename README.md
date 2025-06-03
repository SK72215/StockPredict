
# IntelliStock AI Stock Predictor – Project Summary

## Part 1: End-to-End Development Summary

### 1. Environment Setup
- Installed Homebrew and Python 3.
- Created a virtual environment:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 2. Project Structure
```
StockPredict/
├── app.py
├── portfolio.json
├── ai_stock_predictor/
│   └── predictor.py
└── requirements.txt
```

### 3. Installed Libraries
```bash
pip install streamlit yfinance pandas numpy altair fuzzywuzzy prophet scikit-learn
```

### 4. GitHub Repository
```bash
git init
git remote add origin https://github.com/YourUsername/StockPredict.git
git add .
git commit -m "Initial version"
git push -u origin main
```

### 5. Streamlit Cloud Deployment
- Connected GitHub
- Set `app.py` as entry point
- Added `requirements.txt`
- Deployed and tested

---

## Part 2: Explanation of `app.py`

- **Imports:** Pull in tools like Streamlit, Pandas, Prophet, yFinance, etc.
- **load_company_data():** Reads list of stocks from GitHub.
- **Portfolio Handling:** Saves and loads user's selected stocks in `portfolio.json`.
- **Sidebar UI:** Lets user build a stock portfolio.
- **fetch_data():** Downloads 6 months of stock data using yFinance.
- **AI Prediction:** Uses a custom `AIPredictor` class.
- **Prophet Forecast:** Uses Facebook Prophet to forecast stock price 10 days ahead.
- **Portfolio Historical View:** Line chart showing price over time for each selected stock.
- **Portfolio Normalized View:** Compares how each stock's value changed relative to its start.

--- 
