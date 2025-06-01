# main.py - Starting point for the tool

from predictor import AIPredictor
from tracker import StockTracker
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AI-based Stock Predictor Tool')
    parser.add_argument('--symbols', nargs='+', help='List of stock symbols to track', required=True)
    parser.add_argument('--interval', choices=['hourly', 'daily'], default='daily', help='Prediction interval')
    args = parser.parse_args()

    predictor = AIPredictor()
    tracker = StockTracker(predictor=predictor, interval=args.interval)

    for symbol in args.symbols:
        tracker.track(symbol)

    tracker.run()
