import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
tickers = ["AAPL", "MSFT", "NVDA", "SPY"]

data = yf.download(
    tickers,
    start="2020-01-01",
    end="2026-01-01",
    auto_adjust=True
)

print(data.head())
print("Everything loaded successfully!!!")
