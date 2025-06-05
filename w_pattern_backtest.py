import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import telegram

# ====== 可調整參數區塊 ======
TICKER = "2330.TW"           # Yahoo Finance 代碼
INTERVAL = "60m"             # K 線週期
PERIOD = "600d"              # 取樣天數

# 小型 W 底參數
MIN_ORDER_SMALL = 3          # 極值偵測 window
P1P3_TOL_SMALL = 0.08        # P1-P3 價格容忍度
PULLBACK_LO_SMALL = 0.99     # 拉回下限
PULLBACK_HI_SMALL = 1.01     # 拉回上限
BREAKOUT_PCT_SMALL = 0.005   # 頸線突破百分比

# 大型 W 底參數
MIN_ORDER_LARGE = 24         # 大型極值偵測 window (約一天以上)
P1P3_TOL_LARGE = 0.25        # P1-P3 價格容忍度
PULLBACK_LO_LARGE = 0.95     # 拉回下限
PULLBACK_HI_LARGE = 1.05     # 拉回上限
BREAKOUT_PCT_LARGE = 0.0025  # 頸線突破百分比

# 停利停損與資金參數
STOP_LOSS_PCT = 0.03
TRAILING_PCT  = 0.05
INITIAL_CAPITAL = 100.0

# Telegram 設定 (在執行環境或 GitHub Actions Secrets 裡設定)
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "YOUR_CHAT_ID")
bot = telegram.Bot(token=BOT_TOKEN)

# ====== 從 Yahoo Finance 下載資料 ======
df = yf.download(TICKER, interval=INTERVAL, period=PERIOD)
df.dropna(inplace=True)
close_prices = df["Close"].to_numpy()
high_prices  = df["High"].to_numpy()
low_prices   = df["Low"].to_numpy()

# ====== 偵測 W 底信號 ======
pullback_signals = []
pattern_points   = []

def detect_w(min_idx, max_idx, tol_p1p3, lo, hi, breakout_pct):
    """
    偵測 W 底形態，條件：P1 < P2, P3 < P2；P1 與 P3 價格接近；
    突破、拉回、再突破觸發信號。
    """
    for i in range(1, len(min_idx)):
        p1 = int(min_idx[i - 1])
        p3 = int(min_idx[i])
        mids = max_idx[(max_idx > p1) & (max_idx < p3)]
        if mids.size == 0:
            continue
        p2 = int(mids[-1])

        p1v = float(close_prices[p1])
        p2v = float(close_prices[p2])
        p3v = float(close_prices[p3])

        # 基本形態：P1 < P2 且 P3 < P2
        if not (p1v < p2v and p3v < p2v):
            continue
        # P1-P3 價差容忍度
        if abs(p1v - p3v) / p1v > tol_p1p3:
            continue

        neckline = p2v
        bo_i = p3 + 1
        if bo_i + 4 >= len(close_prices):
            continue

        bo_v = float(close_prices[bo_i])
        pb_v = float(close_prices[bo_i + 2])
        tr_v = float(close_prices[bo_i + 4])

        # 突破條件
        if bo_v <= neckline * (1 + breakout_pct):
            continue
        # 拉回條件
        if not (neckline * lo < pb_v < neckline * hi):
            continue
        # 再破條件
        if tr_v <= pb_v:
            continue

        # 記錄進場索引、信號價格與頸線價格
        pullback_signals.append((bo_i + 4, tr_v, neckline))
        pattern_points.append((p1, p1v, p2, p2v, p3, p3v, bo_i, bo_v, pb_v, tr_v, breakout_pct))

# 小型 W 底
min_idx_small = argrelextrema(close_prices, np.less_equal, order=MIN_ORDER_SMALL)[0]
max_idx_small = argrelextrema(close_prices, np.greater_equal, order=MIN_ORDER_SMALL)[0]
detect_w(min_idx_small, max_idx_small,
         P1P3_TOL_SMALL, PULLBACK_LO_SMALL, PULLBACK_HI_SMALL, BREAKOUT_PCT_SMALL)

# 大型 W 底
min_idx_large = argrelextrema(close_prices, np.less_equal, order=MIN_ORDER_LARGE)[0]
max_idx_large = argrelextrema(close_prices, np.greater_equal, order=MIN_ORDER_LARGE)[0]
detect_w(min_idx_large, max_idx_large,
         P1P3_TOL_LARGE, PULLBACK_LO_LARGE, PULLBACK_HI_LARGE, BREAKOUT_PCT_LARGE)

# ====== 回測（移動停利 + 固定停損） ======
results = []
for entry_idx, entry_price, neckline in pullback_signals:
    entry_time = df.index[entry_idx]
    peak       = entry_price
    exit_idx   = None
    exit_price = None
    result     = None

    # 從進場到觸及停利/停損或結束
    for j in range(1, len(df) - entry_idx):
        high = float(high_prices[entry_idx + j])
        low  = float(low_prices[entry_idx + j])

        if high > peak:
            peak = high
        trail_stop = peak * (1 - TRAILING_PCT)
        fixed_stop = entry_price * (1 - STOP_LOSS_PCT)
        stop_level = max(trail_stop, fixed_stop)

        if low <= stop_level:
            result = "win" if peak > entry_price else "loss"
            exit_price = stop_level
            exit_idx   = entry_idx + j
            break

    # 若未觸及則收盤平倉
    if result is None:
        exit_idx   = len(df) - 1
        exit_price = float(close_prices[exit_idx])
        result     = "win" if exit_price > entry_price else "loss"

    results.append({
        "entry_time": entry_time,
        "entry":      entry_price,
        "exit_time":  df.index[exit_idx],
        "exit":       exit_price,
        "result":     result
    })

# ====== 結果整理 & Telegram 推播 ======
if results:
    results_df = pd.DataFrame(results)
    results_df["profit_pct"] = (results_df["exit"] - results_df["entry"]) / results_df["entry"] * 100

    # 傳送文字訊息
    msg = ["📈 今日 W 底偵測與回測結果："]
    cap = INITIAL_CAPITAL
    for _, row in results_df.iterrows():
        etime = row["entry_time"].strftime("%Y-%m-%d %H:%M")
        xtime = row["exit_time"].strftime("%Y-%m-%d %H:%M")
        pct   = f"{row['profit_pct']:.2f}%"
        cap  *= (1 + row["profit_pct"] / 100)
        msg.append(f"進場: {etime} @ {row['entry']:.2f} → 出場: {xtime} @ {row['exit']:.2f} | 報酬: {pct}")

    cum_pct = (cap / INITIAL_CAPITAL - 1) * 100
    msg.append(f"💰 初始: {INITIAL_CAPITAL:.2f} → 最終: {cap:.2f} | 累積報酬: {cum_pct:.2f}%")
    text = "\n".join(msg)
    bot.send_message(chat_id=CHAT_ID, text=text)

    # 顯示 DataFrame 與圖表
    print(results_df)
    print(text)

else:
    notice = f"⚠️ 今日無 W 底交易信號，共偵測到 {len(pullback_signals)} 個候選點"
    bot.send_message(chat_id=CHAT_ID, text=notice)
    print(notice)

# ====== 繪圖 ======
plt.figure(figsize=(14, 6))
plt.plot(df["Close"], color="gray", alpha=0.5, label="Close")

plotted = set()
# 標記進場
for idx, price, _ in pullback_signals:
    label = "Entry"
    if label not in plotted:
        plt.scatter(df.index[idx], price, marker="^", color="green", label=label)
        plotted.add(label)
    else:
        plt.scatter(df.index[idx], price, marker="^", color="green")

# 標記出場
for rec in results:
    label = "Exit"
    xt = rec["exit_time"]
    xv = rec["exit"]
    if label not in plotted:
        plt.scatter(xt, xv, marker="v", color="red", label=label)
        plotted.add(label)
    else:
        plt.scatter(xt, xv, marker="v", color="red")

# 標記 W 底關鍵點
for p1, p1v, p2, p2v, p3, p3v, bidx, bo, pb, tr, bpct in pattern_points:
    lbl1 = f"P1_{bpct}"
    lbl2 = f"P2_{bpct}"
    lbl3 = f"P3_{bpct}"
    lblN = f"Neck_{bpct}"

    if lbl1 not in plotted:
        plt.scatter(df.index[p1], p1v, color="blue", marker="o", label=lbl1)
        plotted.add(lbl1)
    else:
        plt.scatter(df.index[p1], p1v, color="blue", marker="o")

    if lbl2 not in plotted:
        plt.scatter(df.index[p2], p2v, color="orange", marker="o", label=lbl2)
        plotted.add(lbl2)
    else:
        plt.scatter(df.index[p2], p2v, color="orange", marker="o")

    if lbl3 not in plotted:
        plt.scatter(df.index[p3], p3v, color="blue", marker="o", label=lbl3)
        plotted.add(lbl3)
    else:
        plt.scatter(df.index[p3], p3v, color="blue", marker="o")

    if lblN not in plotted:
        plt.hlines(p2v, df.index[p1], df.index[p3], colors="purple", linestyles="dashed", label=lblN)
        plotted.add(lblN)
    else:
        plt.hlines(p2v, df.index[p1], df.index[p3], colors="purple", linestyles="dashed")

plt.title(f"{TICKER} W-Pattern Strategy")
plt.xlabel("時間")
plt.ylabel("價格")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()
