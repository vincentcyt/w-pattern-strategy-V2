# 文件名：w_pattern_backtest.py

import os
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import telegram  # 记得在 requirements.txt 里加 “python-telegram-bot==13.7”（或同系列最新）

# ====== Telegram Bot 设置 ======
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")    # 存在 GitHub Secrets：TELEGRAM_BOT_TOKEN
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")      # 存在 GitHub Secrets：TELEGRAM_CHAT_ID

# 若 Bot Token 或 Chat ID 没设，直接跳过发送
send_flag = True
if not BOT_TOKEN or not CHAT_ID:
    print("Error: Telegram token or chat_id not configured.")
    send_flag = False

# ====== 参数区（可根据需求再调） ======
TICKER = "2330.TW"
INTERVAL = "60m"
PERIOD  = "600d"        # 最近 600 天

# 小型 W 底参数
MIN_ORDER_SMALL = 3
P1P3_TOL_SMALL  = 0.08
PULLBACK_LO_SMALL, PULLBACK_HI_SMALL = 0.99, 1.01

# 大型 W 底参数
MIN_ORDER_LARGE = 24
P1P3_TOL_LARGE  = 0.25
PULLBACK_LO_LARGE, PULLBACK_HI_LARGE = 0.95, 1.05

# 统一参数
BREAKOUT_PCT    = 0.005
INITIAL_CAPITAL = 100.0
TRAILING_PCT    = 0.05
STOP_PCT        = 0.03


# ====== 下载数据 ======
df = yf.download(TICKER, interval=INTERVAL, period=PERIOD)
df.dropna(inplace=True)
close_prices = df["Close"].to_numpy()
high_prices  = df["High"].to_numpy()
low_prices   = df["Low"].to_numpy()


# ====== W 底检测函数 ======
pullback_signals = []   # 记录所有新信号 (entry_idx, entry_price, neckline)
pattern_points   = []   # 记录画图用的 (p1,p1v,p2,p2v,p3,p3v,bo,pb,tr)

def detect_w(min_idx, max_idx, tol_p1p3, lo, hi):
    """
    找出 W 底。min_idx 和 max_idx 是局部极值的索引；tol_p1p3 是 p1/p3 相似度门槛；
    lo/hi 定义“拉回价位”与颈线的乘数范围。
    """
    for i in range(1, len(min_idx)):
        p1 = int(min_idx[i-1])
        p3 = int(min_idx[i])
        # p2 必须在 p1 和 p3 之间且是局部最高
        mids = max_idx[(max_idx > p1) & (max_idx < p3)]
        if mids.size == 0:
            continue
        p2 = int(mids[-1])

        p1v = close_prices[p1]
        p2v = close_prices[p2]
        p3v = close_prices[p3]
        # 基本结构：p1 < p2 且 p3 < p2
        if not (p1v < p2v and p3v < p2v):
            continue
        # p1/p3 相似度
        if abs(p1v - p3v) / p1v > tol_p1p3:
            continue

        neckline = p2v
        bo_i     = p3 + 1
        if bo_i + 4 >= len(close_prices):
            continue
        bo_v = close_prices[bo_i]
        pb_v = close_prices[bo_i + 2]   # 拉回价
        tr_v = close_prices[bo_i + 4]   # 触发价

        # 进场条件：突破颈线 + 拉回到颈线区间 + 触发价 > 拉回价
        if bo_v <= neckline * (1 + BREAKOUT_PCT):
            continue
        if not (neckline * lo < pb_v < neckline * hi):
            continue
        if tr_v <= pb_v:
            continue

        pullback_signals.append((bo_i + 4, tr_v, neckline))
        pattern_points.append((p1, p1v, p2, p2v, p3, p3v, bo_i, bo_v, pb_v, tr_v))


# 小型 W 底
min_idx_s = argrelextrema(close_prices, np.less_equal, order=MIN_ORDER_SMALL)[0]
max_idx_s = argrelextrema(close_prices, np.greater_equal, order=MIN_ORDER_SMALL)[0]
detect_w(min_idx_s, max_idx_s, P1P3_TOL_SMALL, PULLBACK_LO_SMALL, PULLBACK_HI_SMALL)

# 大型 W 底
min_idx_L = argrelextrema(close_prices, np.less_equal, order=MIN_ORDER_LARGE)[0]
max_idx_L = argrelextrema(close_prices, np.greater_equal, order=MIN_ORDER_LARGE)[0]
detect_w(min_idx_L, max_idx_L, P1P3_TOL_LARGE, PULLBACK_LO_LARGE, PULLBACK_HI_LARGE)


# ====== 回测（无持有期限制：移动止盈 + 固定止损） ======
results = []

for eidx, eprice, neckline in pullback_signals:
    entry_time = df.index[eidx]
    peak       = eprice
    exit_price = None
    exit_idx   = None
    result     = None

    stop_level = eprice * (1 - STOP_PCT)
    for j in range(1, len(df) - eidx):
        high = high_prices[eidx + j]
        low  = low_prices[eidx + j]
        # 更新最高
        if high > peak:
            peak = high
        trail_stop = peak * (1 - TRAILING_PCT)
        actual_stop = max(stop_level, trail_stop)
        if low <= actual_stop:
            exit_price = actual_stop
            exit_idx   = eidx + j
            result     = "win" if peak > eprice else "loss"
            break

    if result is None:
        exit_idx   = len(df) - 1
        exit_price = close_prices[exit_idx]
        result     = "win" if exit_price > eprice else "loss"

    results.append({
        "entry_time": entry_time,
        "entry":      eprice,
        "exit_time":  df.index[exit_idx],
        "exit":       exit_price,
        "result":     result
    })


# ====== 结果整理 & 发送 Telegram 群消息 ======
import datetime

# 1) 列出每笔交易的 entry/exit 时间、价格、profit_pct
results_df = pd.DataFrame(results)
if not results_df.empty:
    results_df["profit_pct"] = (results_df["exit"] - results_df["entry"]) / results_df["entry"] * 100

# 2) 最终资金与累计回报
cap = INITIAL_CAPITAL
for pct in results_df["profit_pct"]:
    cap *= (1 + pct / 100)
cum_ret = (cap / INITIAL_CAPITAL - 1) * 100

# 3) 构造要发送的文本内容
today_str = datetime.datetime.now().strftime("%Y-%m-%d")
if not results_df.empty:
    msg = f"📊 W-Pattern 回测结果 ({today_str})\n"
    msg += "————————————\n"
    for idx, row in results_df.iterrows():
        msg += (f"{idx+1}. Entry: {row['entry_time'].strftime('%Y-%m-%d %H:%M')} @ {row['entry']:.2f}，"
                f"Exit: {row['exit_time'].strftime('%Y-%m-%d %H:%M')} @ {row['exit']:.2f}，"
                f"收益: {row['profit_pct']:.2f}%\n")
    msg += f"————————————\n初始 {INITIAL_CAPITAL:.2f}，最终 {cap:.2f}，累计 {cum_ret:.2f}%"
else:
    msg = f"{today_str} 今日无W底信号。"

print(msg)


# 4) 发送到 Telegram（只有当检测到 BOT_TOKEN & CHAT_ID 且有新信号时才发）
if send_flag:
    bot = telegram.Bot(token=BOT_TOKEN)
    bot.send_message(chat_id=CHAT_ID, text=msg)
