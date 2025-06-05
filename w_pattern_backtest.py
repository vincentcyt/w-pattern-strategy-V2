#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from telegram import Bot

# —— Telegram Bot 相关 —— #
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")
if not BOT_TOKEN or not CHAT_ID:
    raise ValueError("需要在环境变量里设置 BOT_TOKEN 和 CHAT_ID")
bot = Bot(token=BOT_TOKEN)

# ====== 参数区（方便调整） ======
TICKER = "2330.tw"
INTERVAL = "60m"
PERIOD = "600d"

# 小型 W 参数
MIN_ORDER_SMALL     = 3
P1P3_TOL_SMALL      = 0.9
PULLBACK_LO_SMALL   = 0.8
PULLBACK_HI_SMALL   = 1.2

# 大型 W 参数
MIN_ORDER_LARGE     = 200
P1P3_TOL_LARGE      = 0.9
PULLBACK_LO_LARGE   = 0.78
PULLBACK_HI_LARGE   = 1.4

# 统一参数
BREAKOUT_PCT    = 0.00001
INITIAL_CAPITAL = 100.0
TRAILING_PCT    = 0.08
STOP_PCT        = 0.10

# ====== 数据下载 ======
# 注意：从 2024 年起，yfinance.download() 的 auto_adjust 默认已经是 True，如果要关闭请加 auto_adjust=False
df = yf.download(TICKER, interval=INTERVAL, period=PERIOD, auto_adjust=False)
df.dropna(inplace=True)

# 转为 numpy arrays
close_prices = df['Close'].to_numpy()
high_prices  = df['High'].to_numpy()
low_prices   = df['Low'].to_numpy()

# ====== 寻找 W 底信号 ======
pullback_signals = []   # (signal_idx, entry_price, neckline)
pattern_points   = []   # 详细点位，调试时可以用来绘图标注

def detect_w(min_idx, max_idx, tol_p1p3, lo, hi):
    for i in range(1, len(min_idx)):
        p1 = int(min_idx[i - 1])
        p3 = int(min_idx[i])
        # p2 必须是 p1~p3 之间的最大极值
        mids = max_idx[(max_idx > p1) & (max_idx < p3)]
        if mids.size == 0:
            continue
        p2 = int(mids[-1])

        # 取出收盘价
        p1v = float(close_prices[p1].item())
        p2v = float(close_prices[p2].item())
        p3v = float(close_prices[p3].item())

        # 基本形态：两个低 (p1、p3) 均低于中间高点 p2
        if not (p1v < p2v and p3v < p2v):
            continue

        # P1 与 P3 必须“差不多相等”
        if abs(p1v - p3v) / p1v > tol_p1p3:
            continue

        # 颈线价格 = p2v
        neckline = p2v
        bo_i     = p3 + 1  # 突破索引（下一根 K 线上涨）
        if bo_i + 4 >= len(close_prices):
            continue

        bo_v = float(close_prices[bo_i].item())         # 突破价
        pb_v = float(close_prices[bo_i + 2].item())     # 拉回价
        tr_v = float(close_prices[bo_i + 4].item())     # 触发价

        # 突破条件：bo_v 必须大于 颈线*(1+BREAKOUT_PCT)
        if bo_v <= neckline * (1 + BREAKOUT_PCT):
            continue

        # 拉回区间：拉回价必须落在 [neckline*lo, neckline*hi]
        if not (neckline * lo < pb_v < neckline * hi):
            continue

        # 最后触发：tr_v 必须继续往上，tr_v > pb_v
        if tr_v <= pb_v:
            continue

        # 将信号记录下来
        pullback_signals.append((bo_i + 4, tr_v, neckline))
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
    exit_price = None
    exit_idx   = None

    # 从入场后的下一个小时开始 逐根 K 线检查止盈/止损
    for offset in range(1, len(df) - entry_idx):
        h = float(high_prices[entry_idx + offset].item())
        l = float(low_prices[entry_idx + offset].item())
        peak = max(peak, h)

        trail_stop = peak * (1 - TRAILING_PCT)          # 移动止损
        fixed_stop = entry_price * (1 - STOP_PCT)        # 固定止损
        stop_level = max(trail_stop, fixed_stop)

        if l <= stop_level:
            # 触发止损或止盈
            result     = 'win' if peak > entry_price else 'loss'
            exit_price = stop_level
            exit_idx   = entry_idx + offset
            break

    # 如果整个持仓期都没触发止损止盈，就以最后一根 K 线的收盘价对冲平仓
    if result is None:
        exit_idx   = len(df) - 1
        exit_price = float(close_prices[exit_idx].item())
        result     = 'win' if exit_price > entry_price else 'loss'

    results.append({
        'entry_time': entry_time,
        'entry':      entry_price,
        'exit_time':  df.index[exit_idx],
        'exit':       exit_price,
        'result':     result
    })


# ====== 结果展示 ======
if results:
    results_df = pd.DataFrame(results)
    # 计算单次收益百分比
    results_df['profit_pct'] = (results_df['exit'] - results_df['entry']) / results_df['entry'] * 100

    # 构造要发送给 Telegram 的文本
    msg = ""
    for idx, row in results_df.iterrows():
        # 确保 row['entry']、row['exit']、row['profit_pct'] 都是 Python float
        e_price = float(row['entry'])
        x_price = float(row['exit'])
        p_pct   = float(row['profit_pct'])

        msg += (
            f"{idx+1}. Entry: {row['entry_time'].strftime('%Y-%m-%d %H:%M')}  @ {e_price:.2f}，"
            f" Exit: {row['exit_time'].strftime('%Y-%m-%d %H:%M')}  @ {x_price:.2f}，"
            f" Profit: {p_pct:.2f}%\n"
        )

    # 计算累计资金和总回报
    cap = INITIAL_CAPITAL
    for p_pct in results_df['profit_pct']:
        cap *= (1 + float(p_pct) / 100)
    cum_ret = (cap / INITIAL_CAPITAL - 1) * 100
    msg += f"\n初始资金：{INITIAL_CAPITAL:.2f}，最终资金：{cap:.2f}，累计回报：{cum_ret:.2f}%"

    # 发送给 Telegram
    bot.send_message(chat_id=CHAT_ID, text=msg)
else:
    # 如果今日日内没有任何信号，就发一句“今日无讯号”
    bot.send_message(chat_id=CHAT_ID, text="📊 今日无 W 底信号")



# ====== 绘图（非必须，仅供本地调试） ======
# 如果不需要绘图，可以注释掉下面这一段
if pattern_points:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Close'], color='gray', alpha=0.5, label='Close')

    plotted = set()
    def safe_label(lbl):
        if lbl in plotted:
            return "_nolegend_"
        plotted.add(lbl)
        return lbl

    # 标注进/出场点
    for tr in results:
        ax.scatter(tr['entry_time'], tr['entry'], marker='^', c='green', label=safe_label('Entry'))
        ax.scatter(tr['exit_time'],  tr['exit'],  marker='v', c='red',   label=safe_label('Exit'))

    # 标注 W 底结构点（仅示例，不发送到 Telegram）
    for p1, p1v, p2, p2v, p3, p3v, bo_i, bo_v, pb_v, tr_v, tol in pattern_points:
        ax.scatter(df.index[p1], p1v, c='blue',  marker='o', label=safe_label('P1'))
        ax.scatter(df.index[p3], p3v, c='blue',  marker='o', label=safe_label('P3'))
        ax.scatter(df.index[p2], p2v, c='orange',marker='o', label=safe_label('P2'))
        ax.hlines(p2v, df.index[p1], df.index[p3], colors='purple', linestyle='dashed', label=safe_label('Neckline'))
        ax.scatter(df.index[bo_i], bo_v, c='cyan',  marker='x', label=safe_label('Breakout'))
        ax.scatter(df.index[bo_i+2], pb_v, c='magenta', marker='x', label=safe_label('Pullback'))
        ax.scatter(df.index[bo_i+4], tr_v, c='lime', marker='x', label=safe_label('Trigger'))

    ax.set_title(f"{TICKER} W-Pattern Strategy")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend(loc="best")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("w_pattern_plot.png")  # 如果想保存图片，也可以
    # plt.show()
