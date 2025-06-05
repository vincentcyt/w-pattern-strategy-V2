import yfinance as yf
df = yf.download("2330.TW", period="60d", interval="10m", auto_adjust=False)
print(df.head(), df.tail())
