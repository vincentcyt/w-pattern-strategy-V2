#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import telegram
from telegram import Bot

# ====== 环境变量（请在 GitHub Secrets 或本地环境中设置这两个） ======
# BOT_TOKEN: 你从 @BotFather 那里得到的 Bot 令牌
# CHAT_ID:  你要发送消息的 Telegram 聊天 ID（可以是私聊 ID 或 群组 ID）
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")

if not BOT_TOKEN or not CHAT_ID:
    raise ValueError("需要在环境变量里设置 BOT_TOKEN 和 CHAT_ID")

bot = telegram.Bot(token=BOT_TOKEN)

bot.send_message(chat_id=CHAT_ID, text=f"Morning~~")

# ====== 参数区（方便调整） ======
TICKER           = "2330.tw"   # Yahoo Finance 上的代码
INTERVAL         = "60m"       # 数据周期
PERIOD           = "600d"      # 数据长度

# 小型 W 参数
MIN_ORDER_SMALL  = 3           # 小型 W 极值识别窗口
P1P3_TOL_SMALL   = 0.15        # P1 与 P3 相似度容差（小型 W）
PULLBACK_LO_SMALL, PULLBACK_HI_SMALL = 0.99, 1.01  # 小型 W 拉回区域

# 大型 W 参数
MIN_ORDER_LARGE  = 200         # 大型 W 极值识别窗口 (约一天以上周期)
P1P3_TOL_LARGE   = 0.25        # P1 与 P3 相似度容差（大型 W）
PULLBACK_LO_LARGE, PULLBACK_HI_LARGE = 0.95, 1.05  # 大型 W 拉回区域（放宽）

# 统一参数
BREAKOUT_PCT     = 0.001       # 突破颈线百分比
INITIAL_CAPITAL  = 100.0       # 初始资金
TRAILING_PCT     = 0.07        # 移动止盈百分比
STOP_PCT         = 0.03        # 固定止损百分比

# ====== 数据下载 ======
# 注意：为保持与过去版本一致，这里将 auto_adjust 显式设为 False
df = yf.download(
    TICKER,
    interval=INTERVAL,
    period=PERIOD,
    auto_adjust=False,
    progress=False
)
df.dropna(inplace=True)

# 转为 numpy arrays
close_prices = df['Close'].to_numpy()
high_prices  = df['High'].to_numpy()
low_prices   = df['Low'].to_numpy()

# ====== 寻找 W 底信号 ======
pullback_signals = []   # 存放 (触发索引, 触发价, 颈线价)
pattern_points   = []   # 存放 (p1_idx, p1_val, p2_idx, p2_val, p3_idx, p3_val, bo_idx, bo_val, pb_val, tr_val, tol)

def detect_w(min_idx, max_idx, tol_p1p3, lo, hi):
    """
    Detect W patterns given extrema indices and tolerances.
    lo/hi define pullback zone multipliers for neckline.
    """
    for i in range(1, len(min_idx)):
        p1 = int(min_idx[i-1])
        p3 = int(min_idx[i])
        # p2 必须是 p1 和 p3 之间的最高点
        mids = max_idx[(max_idx > p1) & (max_idx < p3)]
        if mids.size == 0:
            continue
        p2 = int(mids[-1])

        # 提取价格为 Python float
        p1v = float(close_prices[p1])
        p2v = float(close_prices[p2])
        p3v = float(close_prices[p3])

        # 基本结构检查：P1 < P2 且 P3 < P2
        if not (p1v < p2v and p3v < p2v):
            continue

        # P1-P3 相似度检验
        if abs(p1v - p3v) / p1v > tol_p1p3:
            continue

        # 颈线价格
        neckline = p2v
        bo_i     = p3 + 1
        if bo_i + 4 >= len(close_prices):
            continue

        bo_v = float(close_prices[bo_i])       # 突破价格
        pb_v = float(close_prices[bo_i + 2])   # 拉回价格
        tr_v = float(close_prices[bo_i + 4])   # 触发价格

        # 进场条件：
        # 1) 突破价 > 颈线 * (1 + BREAKOUT_PCT)
        if bo_v <= neckline * (1 + BREAKOUT_PCT):
            continue
        # 2) 拉回价 在 [颈线*lo, 颈线*hi] 之间
        if not (neckline * lo < pb_v < neckline * hi):
            continue
        # 3) 触发价 > 拉回价
        if tr_v <= pb_v:
            continue

        # 如果满足所有条件，就把信号记录下来
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
    exit_idx   = None

    # 持有期直到移动止盈或固定止损触发
    for j in range(1, len(df) - entry_idx):
        high = float(high_prices[entry_idx + j])
        low  = float(low_prices[entry_idx + j])
        peak = max(peak, high)

        trail_stop = peak * (1 - TRAILING_PCT)
        fixed_stop = entry_price * (1 - STOP_PCT)
        stop_level = max(trail_stop, fixed_stop)

        if low <= stop_level:
            # 如果触及止损／止盈线，记录并跳出
            result     = 'win' if peak > entry_price else 'loss'
            exit_price = stop_level
            exit_idx   = entry_idx + j
            break

    # 如果持有到最后也没触发，就在最后一个收盘价平仓
    if result is None:
        exit_idx   = len(df) - 1
        exit_price = float(close_prices[exit_idx])
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
    results_df['profit_pct'] = (results_df['exit'] - results_df['entry']) / results_df['entry'] * 100

    # 输出每笔交易明细
    print("\n===== 每笔交易明细 =====")
    print(results_df[['entry_time', 'entry', 'exit_time', 'exit', 'result', 'profit_pct']])

    # 计算资金演变
    cap = INITIAL_CAPITAL
    for pct in results_df['profit_pct']:
        cap *= (1 + float(pct) / 100)
    cum_ret = (cap / INITIAL_CAPITAL - 1) * 100

    print(f"\n初始资金: {INITIAL_CAPITAL:.2f}")
    print(f"最终资金: {cap:.2f}")
    print(f"累计回报: {cum_ret:.2f}%\n")

    # ====== 发送 Telegram 消息 ======
    msg = f"📊 {TICKER} W 底策略回测结果：\n\n"
    for idx, row in results_df.iterrows():
        entry_t_str = row['entry_time'].strftime('%Y-%m-%d %H:%M')
        exit_t_str  = row['exit_time'].strftime('%Y-%m-%d %H:%M')
        entry_p     = float(row['entry'])
        exit_p      = float(row['exit'])
        profit_pct  = float(row['profit_pct'])
        msg += (
            f"{idx+1}. Entry: {entry_t_str} @ {entry_p:.2f}，"
            f"Exit: {exit_t_str} @ {exit_p:.2f}，"
            f"Profit: {profit_pct:.2f}%\n"
        )
    msg += f"\n初始资金: {INITIAL_CAPITAL:.2f}，最终资金: {cap:.2f}，累计回报: {cum_ret:.2f}%"
    bot.send_message(chat_id=CHAT_ID, text=msg)

else:
    print("⚠️ 无交易信号，共 0 个信号")
    bot.send_message(chat_id=CHAT_ID, text=f"⚠️ {TICKER} 在给定期间内未检测到 W 底信号。")


# ====== 绘图 ======
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(df['Close'], color='gray', alpha=0.5, label='Close Price')

plotted = set()
def safe_label(lbl):
    if lbl in plotted: 
        return '_nolegend_'
    plotted.add(lbl)
    return lbl

# 标注每笔交易：进/出场
if results:
    for tr in results:
        ax.scatter(tr['entry_time'], tr['entry'], marker='^', c='green', s=80, label=safe_label('Entry'))
        ax.scatter(tr['exit_time'],  tr['exit'],  marker='v', c='red',   s=80, label=safe_label('Exit'))

# 标注结构点：P1, P2, P3, Neckline, Breakout, Pullback, Trigger
for p1, p1v, p2, p2v, p3, p3v, bo, bo_v, pb_v, tr_v, tol in pattern_points:
    color = 'blue' if tol == P1P3_TOL_SMALL else 'darkblue'
    ax.scatter(df.index[p1], p1v, c=color, marker='o', s=50, label=safe_label('P1'))
    ax.scatter(df.index[p3], p3v, c=color, marker='o', s=50, label=safe_label('P3'))
    ax.scatter(df.index[p2], p2v, c='orange', marker='o', s=50, label=safe_label('P2'))
    ax.hlines(p2v, df.index[p1], df.index[p3], colors='purple', linestyles='dashed', label=safe_label('Neckline'))
    ax.scatter(df.index[bo],    bo_v, c='cyan',    marker='x', s=70, label=safe_label('Breakout'))
    ax.scatter(df.index[bo + 2], pb_v, c='magenta', marker='x', s=70, label=safe_label('Pullback'))
    ax.scatter(df.index[bo + 4], tr_v, c='lime',    marker='x', s=70, label=safe_label('Trigger'))

ax.set_title(f"{TICKER} W-Pattern Strategy 回测示意图", fontsize=16)
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Price', fontsize=12)
ax.legend(loc='best', fontsize=10)
ax.grid(True)
plt.tight_layout()

# 把图保存到本地，GitHub Actions 环境下可以查看 artifacts
output_plot = "w_pattern_backtest_plot.png"
plt.savefig(output_plot)
print(f"已将回测图保存为: {output_plot}")
