import yfinance as yf

# 定义标的与数据区间
TICKER = "2330.TW"
PERIOD = "5d"   # 过去 5 天

# 从 Yahoo Finance 下载数据
df = yf.download(TICKER, period=PERIOD)

# 显示结果
print(df)
