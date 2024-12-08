import yfinance as yf

# Get data for Tata Steel (TATASTEEL.NS)

tata_steel = yf.Ticker("TATASTEEL.NS")

# Download historical data

tata_steel_data = tata_steel.history(period="10y")  

# Print the first few rows of the data

print(tata_steel_data.head())

'''
import yfinance as yf
import pandas as pd
# Fetch Stock Data for the Last 10 Years
def fetch_historical_data(symbols):
    for symbol in symbols:
        print(f"Fetching 10 years of data for {symbol}")
        stock = yf.Ticker(symbol)
        hist = stock.history(period="10y", interval="1d")
        hist.reset_index(inplace=True)
        hist["Symbol"] = symbol

STOCK_SYMBOLS = ["TATASTEEL.NS", "TCS.NS", "RELIANCE.NS"]
data = fetch_historical_data(STOCK_SYMBOLS)
print(data.head())
'''


