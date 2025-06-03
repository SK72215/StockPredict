import requests
import pandas as pd
from bs4 import BeautifulSoup

# Placeholder for Reddit logic (mocked for now)
def get_reddit_trending():
    # This should be replaced with Reddit API or Pushshift integration if needed
    return ["TSLA", "NVDA", "AMD"]

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

def get_combined_trending():
    reddit = get_reddit_trending()
    stocktwits = get_stocktwits_trending()

    all_sources = reddit + stocktwits
    weighted_counts = pd.Series(all_sources).value_counts()
    top_trending = weighted_counts[weighted_counts > 1].index.tolist()

    # If overlap is low, still return top few
    if not top_trending:
        top_trending = weighted_counts.index.tolist()[:5]

    return top_trending
