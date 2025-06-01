# tracker.py - Stock tracking logic

from time import sleep

class StockTracker:
    def __init__(self, predictor, interval='daily'):
        self.predictor = predictor
        self.interval = interval
        self.symbols = []

    def track(self, symbol: str):
        print(f"[INFO] Tracking {symbol} for {self.interval} predictions")
        self.symbols.append(symbol)

    def run(self):
        print("[INFO] Running stock predictions...")
        for symbol in self.symbols:
            result = self.predictor.predict(symbol, self.interval)
            print(f"[RESULT] {result['symbol']}: {result['prediction']} ({result['confidence']*100}%)")
        print("[INFO] Prediction run complete")
