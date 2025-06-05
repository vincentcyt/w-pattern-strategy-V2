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

# ———— 先检查环境变量 BOT_TOKEN 和 CHAT_ID ———— #
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")
print(f"[DEBUG] BOT_TOKEN is [{'set' if BOT_TOKEN else 'NOT set'}]")
print(f"[DEBUG] CHAT_ID   is [{'set' if CHAT_ID else 'NOT set'}]")

if not BOT_TOKEN or not CHAT_ID:
    print("❌ ERROR: 必须在环境变量里设置 BOT_TOKEN 和 CHAT_ID，程序退出。")
    sys.exit(1)

bot = Bot(token=BOT_TOKEN)

# ====== 参数区（方便调整） ======
TICKER = "2330.TW"      # 注意改成大写
INTERVAL = "60m"
PERIOD   = "600d"

# 小型 W 参数
MIN_ORDER_SMALL   = 3
P1P3_TOL_SMALL    = 0.9
PULLBACK_LO_SMALL = 0.8
PULLBACK_HI_SMALL = 1.2

# 大型 W 参数
MIN_ORDER_LARGE   = 200
P1P3_TOL_LARGE    = 0.9
PULLBACK_LO_LARGE = 0.78
PULLBACK_HI_LARGE = 1.4

# 统一参数
BREAKOUT_PCT    = 0.00001
INITIAL_CAPITAL = 100.0
TRAILING_PCT    = 0.08
STOP_PCT        = 0.10

# ====== 数据下载 ======
# 注意：yfinance download 默认 auto_adjust=True，如果想拿未复权价格可以 auto_adjust=False
df = yf.download(TICKER, interval=INTERVAL, period=PERIOD, auto_adjust=False)
if df.empty:
    # 如果下载失败，直接通知并退出
    bot.send_message(chat_id=CHAT_ID, text=f"❌ 无法获取 {TICKER} 的数据，请检查符号或网络。")
    sys.exit(0)

df.dropna(inplace=True)

close_prices = df['Close'].to_numpy()
high_prices  = df['High'].to_numpy()
low_prices   = df['Low'].to_numpy()

# ====== 寻找 W 底信号 ======
pullback_signals = []
pattern_points   = []

def detect_w(min_idx, max_idx, tol_p1p3, lo, hi):
    """
    检测所有符合 W 底形态的（触发点索引, 触发价, 颈线价）。
    min_idx: 所有局部极小值（P1, P3）的索引数组
    max_idx: 所有局部极大值（P2）的索引数组
    tol_p1p3: P1 与 P3 允许的价格相差比例
    lo, hi: 拉回价格必须在 [lo * 颈线, hi * 颈线] 之间
    """
    for i in range(1, len(min_idx)):
        p1 = int(min_idx[i - 1])
        p3 = int(min_idx[i])
        # p2 必须是 p1 与 p3 之间的最后一个局部极大值
        mids = max_idx[(max_idx > p1) & (max_idx < p3)]
        if mids.size == 0:
            continue
        p2 = int(mids[-1])

        p1v = float(close_prices[p1].item())
        p2v = float(close_prices[p2].item())
        p3v = float(close_prices[p3].item())
        # 基本形：P1 < P2 且 P3 < P2
        if not (p1v < p2v and p3v < p2v):
            continue
        # P1 与 P3 价格要在 tol_p1p3 范围内
        if abs(p1v - p3v) / p1v > tol_p1p3:
            continue

        # 颈线价格
        neckline = p2v
        # 突破点为 p3 + 1
        bo_i = p3 + 1
        # 如果不足 4 根 K 线来检验拉回和触发，就跳过
        if bo_i + 4 >= len(close_prices):
            continue

        bo_v = float(close_prices[bo_i].item())       # 突破后第 1 根
        pb_v = float(close_prices[bo_i + 2].item())   # 突破后第 3 根
        tr_v = float(close_prices[bo_i + 4].item())   # 突破后第 5 根

        # 进场条件检查
        # 1) 突破点要 > 颈线 * (1 + BREAKOUT_PCT)
        if bo_v <= neckline * (1 + BREAKOUT_PCT):
            continue
        # 2) 拉回要在 [lo*颈线, hi*颈线]
        if not (neckline * lo < pb_v < neckline * hi):
            continue
        # 3) 触发点要高于拉回点
        if tr_v <= pb_v:
            continue

        pullback_signals.append((bo_i + 4, tr_v, neckline))
        pattern_points.append((p1, p1v, p2, p2v, p3, p3v, bo_i, bo_v, pb_v, tr_v, tol_p1p3))

# —— 找小型 W —— #
min_idx_small = argrelextrema(close_prices, np.less_equal, order=MIN_ORDER_SMALL)[0]
max_idx_small = argrelextrema(close_prices, np.greater_equal, order=MIN_ORDER_SMALL)[0]
detect_w(min_idx_small, max_idx_small, P1P3_TOL_SMALL, PULLBACK_LO_SMALL, PULLBACK_HI_SMALL)

# —— 找大型 W —— #
min_idx_large = argrelextrema(close_prices, np.less_equal, order=MIN_ORDER_LARGE)[0]
max_idx_large = argrelextrema(close_prices, np.greater_equal, order=MIN_ORDER_LARGE)[0]
detect_w(min_idx_large, max_idx_large, P1P3_TOL_LARGE, PULLBACK_LO_LARGE, PULLBACK_HI_LARGE)

# ====== 回测逻辑 ======
results = []
for entry_idx, entry_price, neckline in pullback_signals:
    entry_time = df.index[entry_idx]
    peak       = entry_price
    result     = None
    exit_price = None
    exit_idx   = None

    # 每根 K 线检查一次是否触发止盈/止损
    for offset in range(1, len(df) - entry_idx):
        high = float(high_prices[entry_idx + offset].item())
        low  = float(low_prices[entry_idx + offset].item())
        peak = max(peak, high)

        trail_stop = peak * (1 - TRAILING_PCT)            # 移动止盈价
        fixed_stop = entry_price * (1 - STOP_PCT)         # 固定止损价
        stop_level = max(trail_stop, fixed_stop)

        if low <= stop_level:
            # 触发止盈/止损
            result     = 'win' if peak > entry_price else 'loss'
            exit_price = stop_level
            exit_idx   = entry_idx + offset
            break

    # 如果整个持有期都没触发止盈/止损，则最后一根 K 线收盘平仓
    if result is None:
        exit_idx   = len(df) - 1
        exit_price = float(close_prices[exit_idx].item())
        result     = 'win' if exit_price > entry_price else 'loss'

    results.append({
        'entry_time': entry_time,
        'entry':      float(entry_price),
        'exit_time':  df.index[exit_idx],
        'exit':       float(exit_price),
        'result':     result
    })

# ====== 构造 DataFrame，计算收益率 ======
if results:
    results_df = pd.DataFrame(results)
    # 强制转换为 datetime（如果从 GitHub Actions 传入的是字符串，也能转换）
    results_df['entry_time'] = pd.to_datetime(results_df['entry_time'])
    results_df['exit_time']  = pd.to_datetime(results_df['exit_time'])
    # 计算 profit_pct
    results_df['profit_pct'] = (results_df['exit'] - results_df['entry']) / results_df['entry'] * 100
    results_df.sort_values(by='entry_time', inplace=True)
    results_df.reset_index(drop=True, inplace=True)
else:
    # 仍然要保留这几列，以避免空 DataFrame 时后续用到这些列报错
    results_df = pd.DataFrame(columns=['entry_time','entry','exit_time','exit','result','profit_pct'])


# ====== 判断“今天”信号，并发送 Telegram 消息 ======
today_utc_date = pd.Timestamp.utcnow().date()

# 先初始化一个空的 DataFrame，以防后续逻辑中未定义 results_today 变量
results_today = pd.DataFrame()

if not results_df.empty:
    # 使用 .dt 访问器前，先确保 entry_time 已经是 datetime 类型
    results_today = results_df.loc[
        results_df['entry_time'].dt.tz_convert('UTC').dt.date == today_utc_date
    ]

if not results_today.empty:
    # 当天有信号
    msg_lines = ["📈 今日新增 W 底信号："]
    for idx, row in results_today.iterrows():
        e_time = row['entry_time'].strftime('%Y-%m-%d %H:%M')
        x_time = row['exit_time'].strftime('%Y-%m-%d %H:%M')
        e_price = float(row['entry'])
        x_price = float(row['exit'])
        p_pct   = float(row['profit_pct'])
        msg_lines.append(
            f"{idx+1}. Entry: {e_time} @ {e_price:.2f}  →  Exit: {x_time} @ {x_price:.2f}  Profit: {p_pct:.2f}%"
        )

    # 计算当日累计收益（从 INITIAL_CAPITAL 开始，假设只做今日信号）
    cap = INITIAL_CAPITAL
    for p_pct in results_today['profit_pct']:
        cap *= (1 + float(p_pct)/100)
    cum_ret_today = (cap / INITIAL_CAPITAL - 1) * 100
    msg_lines.append(f"💰 今日交易累计回报：{cum_ret_today:.2f}%")

    final_msg = "\n".join(msg_lines)
    bot.send_message(chat_id=CHAT_ID, text=final_msg)

else:
    # 今日无信号，先发“今日无信号”
    bot.send_message(chat_id=CHAT_ID, text="📊 今日无 W 底信号。")

    # 如果历史上也有信号，就把最近一次发出来
    if not results_df.empty:
        last = results_df.iloc[-1]
        e_time = last['entry_time'].strftime('%Y-%m-%d %H:%M')
        x_time = last['exit_time'].strftime('%Y-%m-%d %H:%M')
        e_price = float(last['entry'])
        x_price = float(last['exit'])
        p_pct   = float(last['profit_pct'])
        hist_msg = (
            f"➡️ 最近一次进/出场信号：\n"
            f"Entry: {e_time} @ {e_price:.2f}\n"
            f"Exit : {x_time} @ {x_price:.2f}\n"
            f"Profit: {p_pct:.2f}%"
        )
        bot.send_message(chat_id=CHAT_ID, text=hist_msg)
    else:
        # 历史也没信号
        bot.send_message(chat_id=CHAT_ID, text="⚠️ 历史数据里也没有任何 W 底信号。")


# ====== （可选）画图部分，仅供本地或调试时查看 ======
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

    # 标注 W 底结构
    for p1, p1v, p2, p2v, p3, p3v, bo_i, bo_v, pb_v, tr_v, tol in pattern_points:
        ax.scatter(df.index[p1], p1v, c='blue',   marker='o', label=safe_label('P1'))
        ax.scatter(df.index[p3], p3v, c='blue',   marker='o', label=safe_label('P3'))
        ax.scatter(df.index[p2], p2v, c='orange', marker='o', label=safe_label('P2'))
        ax.hlines(p2v, df.index[p1], df.index[p3], colors='purple', linestyle='dashed', label=safe_label('Neckline'))
        ax.scatter(df.index[bo_i], bo_v, c='cyan',    marker='x', label=safe_label('Breakout'))
        ax.scatter(df.index[bo_i+2], pb_v, c='magenta',marker='x', label=safe_label('Pullback'))
        ax.scatter(df.index[bo_i+4], tr_v, c='lime',    marker='x', label=safe_label('Trigger'))

    ax.set_title(f"{TICKER} W-Pattern Strategy")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend(loc="best")
    ax.grid(True)
    plt.tight_layout()

    # 如果想把图存下来并在 Actions 中上传 artifact，可以在这里取消注释：
    # plt.savefig("w_pattern_plot.png")
    # plt.close()
