#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from telegram import Bot

 ====== 参数区（方便调整） ======
TICKER = "2330.TW"
INTERVAL = "60m"        # 数据周期
PERIOD = "600d"         # 数据长度

# 小型 W 参数
MIN_ORDER_SMALL = 3       # 小型 W 极值识别窗口
P1P3_TOL_SMALL = 0.9     # P1 与 P3 相似度容差（小型 W）
PULLBACK_LO_SMALL, PULLBACK_HI_SMALL = 0.8, 1.2  # 小型 W 拉回区域

# 大型 W 参数
MIN_ORDER_LARGE = 200      # 大型 W 极值识别窗口 (约一天以上周期)
P1P3_TOL_LARGE = 0.9     # P1 与 P3 相似度容差（大型 W）
PULLBACK_LO_LARGE, PULLBACK_HI_LARGE = 0.78, 1.4  # 大型 W 拉回区域（放宽）

# 统一参数
BREAKOUT_PCT    = 0.00001      # 突破颈线百分比
INITIAL_CAPITAL = 100.0      # 初始资金
TRAILING_PCT    = 0.08       # 移动止盈百分比
STOP_PCT        = 0.1       # 固定止损百分比

# ====== 数据下载 ======
df = yf.download(TICKER, interval=INTERVAL, period=PERIOD)
df.dropna(inplace=True)
# 转为 numpy arrays
close_prices = df['Close'].to_numpy()
high_prices  = df['High'].to_numpy()
low_prices   = df['Low'].to_numpy()

# ====== 寻找 W 底信号 ======
pullback_signals = []
pattern_points   = []

def detect_w(min_idx, max_idx, tol_p1p3, lo, hi):
    """
    Detect W patterns given extrema indices and tolerances.
    lo/hi define pullback zone multipliers for neckline.
    """
    for i in range(1, len(min_idx)):
        p1 = int(min_idx[i-1])
        p3 = int(min_idx[i])
        # p2 must be highest between p1 and p3
        mids = max_idx[(max_idx > p1) & (max_idx < p3)]
        if mids.size == 0:
            continue
        p2 = int(mids[-1])
        # extract values as python floats
        p1v = close_prices[p1].item()
        p2v = close_prices[p2].item()
        p3v = close_prices[p3].item()
        # 基本形态检查
        if not (p1v < p2v and p3v < p2v):
            continue
        # P1-P3 相似度
        if abs(p1v - p3v) / p1v > tol_p1p3:
            continue
        # 颈线与信号点
        neckline = p2v
        bo_i    = p3 + 1
        if bo_i + 4 >= len(close_prices):
            continue
        bo_v = close_prices[bo_i].item()
        pb_v = close_prices[bo_i+2].item()
        tr_v = close_prices[bo_i+4].item()
        # 进场条件
        if bo_v <= neckline * (1 + BREAKOUT_PCT):
            continue
        if not (neckline * lo < pb_v < neckline * hi):
            continue
        if tr_v <= pb_v:
            continue
        pullback_signals.append((bo_i+4, tr_v, neckline))
        pattern_points.append((p1, p1v, p2, p2v, p3, p3v, bo_i, bo_v, pb_v, tr_v, tol_p1p3))

# 小型 W
min_idx_small = argrelextrema(close_prices, np.less_equal, order=MIN_ORDER_SMALL)[0]
max_idx_small = argrelextrema(close_prices, np.greater_equal, order=MIN_ORDER_SMALL)[0]
detect_w(min_idx_small, max_idx_small, P1P3_TOL_SMALL, PULLBACK_LO_SMALL, PULLBACK_HI_SMALL)
# 大型 W
min_idx_large = argrelextrema(close_prices, np.less_equal, order=MIN_ORDER_LARGE)[0]
max_idx_large = argrelextrema(close_prices, np.greater_equal, order=MIN_ORDER_LARGE)[0]
detect_w(min_idx_large, max_idx_large, P1P3_TOL_LARGE, PULLBACK_LO_LARGE, PULLBACK_HI_LARGE)

# ====== 回测 ======
results = []
for entry_idx, entry_price, neckline in pullback_signals:
    entry_time = df.index[entry_idx]
    peak       = entry_price
    result     = None
    exit_idx   = None
    # 持有期直到止盈/止损
    for j in range(1, len(df) - entry_idx):
        high = high_prices[entry_idx+j].item()
        low  = low_prices[entry_idx+j].item()
        peak = max(peak, high)
        trail_stop = peak * (1 - TRAILING_PCT)
        fixed_stop = entry_price * (1 - STOP_PCT)
        stop_level = max(trail_stop, fixed_stop)
        if low <= stop_level:
            result     = 'win' if peak > entry_price else 'loss'
            exit_price = stop_level
            exit_idx   = entry_idx + j
            break
    # 未触发则收盘平仓
    if result is None:
        exit_idx   = len(df) - 1
        exit_price = close_prices[exit_idx].item()
        result     = 'win' if exit_price > entry_price else 'loss'
    results.append({
        'entry_time': entry_time,
        'entry':      entry_price,
        'exit_time':  df.index[exit_idx],
        'exit':       exit_price,
        'result':     result
    })

# ====== 结果展示 ======
results_df = pd.DataFrame(results)
if not results_df.empty:
    results_df['profit_pct'] = (results_df['exit'] - results_df['entry']) / results_df['entry'] * 100
    print(results_df)
    cap = INITIAL_CAPITAL
    for pct in results_df['profit_pct']:
        cap *= (1 + float(pct)/100)
    cum_ret = (cap/INITIAL_CAPITAL - 1) * 100
    print(f"初始 {INITIAL_CAPITAL:.2f}，最终 {cap:.2f}，累积 {cum_ret:.2f}%")
else:
    print(f"⚠️ 无交易信号，共 {len(pullback_signals)} 个信号")

# ====== 绘图 ======
fig, ax = plt.subplots(figsize=(15,6))
ax.plot(df['Close'], color='gray', alpha=0.5, label='Close')
plotted = set()
def safe_label(lbl):
    if lbl in plotted: return '_nolegend_'
    plotted.add(lbl)
    return lbl
# 标注进/出场
for tr in results:
    ax.scatter(tr['entry_time'], tr['entry'], marker='^', c='green', label=safe_label('Entry'))
    ax.scatter(tr['exit_time'],  tr['exit'],  marker='v', c='red',   label=safe_label('Exit'))
# 标注结构

ax.set_title(f"{TICKER} W-Pattern Strategy")
ax.set_xlabel('Time'); ax.set_ylabel('Price')
ax.legend(loc='best'); ax.grid(True); plt.tight_layout(); plt.show()
