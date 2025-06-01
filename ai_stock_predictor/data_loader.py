# 
# Will contain  logic for historical data loading

# data_loader.py - Historical & live data collection

def fetch_historical_data(symbol, interval='daily'):
    """
    Placeholder for loading historical stock data.
    Args:
        symbol (str): Stock ticker symbol.
        interval (str): 'daily' or 'hourly'
    Returns:
        list: Dummy data series
    """
    return [100 + i for i in range(10)]  # Dummy stock prices


def fetch_news(symbol):
    """
    Placeholder for fetching news headlines for a stock.
    Args:
        symbol (str): Stock ticker symbol.
    Returns:
        list: Dummy headlines
    """
    return [f"Headline {i} for {symbol}" for i in range(3)]

