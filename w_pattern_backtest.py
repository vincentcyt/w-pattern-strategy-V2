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

# —— 调试：打印环境变量是否存在 —— #
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")
print(f"[DEBUG] BOT_TOKEN is [{'set' if BOT_TOKEN else 'NOT set'}]")
print(f"[DEBUG] CHAT_ID   is [{'set' if CHAT_ID else 'NOT set'}]")

if not BOT_TOKEN or not CHAT_ID:
    print("❌ ERROR: 环境变量 BOT_TOKEN 或 CHAT_ID 不存在，程序退出。")
    sys.exit(1)

# 初始化 Telegram Bot
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
df = yf.download(TICKER, interval=INTERVAL, period=PERIOD, auto_adjust=False)
df.dropna(inplace=True)

# 转为 numpy arrays
close_prices = df['Close'].to_numpy()
high_prices  = df['High'].to_numpy()
low_prices   = df['Low'].to_numpy()

# ====== 寻找 W 底信号 ======
pullback_signals = []   # (entry_idx, entry_price, neckline)
pattern_points   = []

def detect_w(min_idx, max_idx, tol_p1p3, lo, hi):
    for i in range(1, len(min_idx)):
        p1 = int(min_idx[i-1]); p3 = int(min_idx[i])
        mids = max_idx[(max_idx > p1) & (max_idx < p3)]
        if mids.size == 0: continue
        p2 = int(mids[-1])
        p1v, p2v, p3v = float(close_prices[p1]), float(close_prices[p2]), float(close_prices[p3])
        if not (p1v < p2v and p3v < p2v): continue
        if abs(p1v - p3v)/p1v > tol_p1p3: continue
        neckline = p2v; bo_i = p3 + 1
        if bo_i + 4 >= len(close_prices): continue
        bo_v = float(close_prices[bo_i])
        pb_v = float(close_prices[bo_i+2])
        tr_v = float(close_prices[bo_i+4])
        if bo_v <= neckline * (1 + BREAKOUT_PCT): continue
        if not (neckline * lo < pb_v < neckline * hi): continue
        if tr_v <= pb_v: continue

        pullback_signals.append((bo_i+4, tr_v, neckline))
        pattern_points.append((p1,p1v,p2,p2v,p3,p3v,bo_i,bo_v,pb_v,tr_v,tol_p1p3))

# 小型 W
min_idx_small = argrelextrema(close_prices, np.less_equal, order=MIN_ORDER_SMALL)[0]
max_idx_small = argrelextrema(close_prices, np.greater_equal, order=MIN_ORDER_SMALL)[0]
detect_w(min_idx_small, max_idx_small, P1P3_TOL_SMALL, PULLBACK_LO_SMALL, PULLBACK_HI_SMALL)

# 大型 W
min_idx_large = argrelextrema(close_prices, np.less_equal, order=MIN_ORDER_LARGE)[0]
max_idx_large = argrelextrema(close_prices, np.greater_equal, order=MIN_ORDER_LARGE)[0]
detect_w(min_idx_large, max_idx_large, P1P3_TOL_LARGE, PULLBACK_LO_LARGE, PULLBACK_HI_LARGE)

# ====== 回测 ======
closed_trades = []
open_trades   = []

for entry_idx, entry_price, neckline in pullback_signals:
    entry_time = df.index[entry_idx]
    peak       = entry_price
    exit_price = None
    exit_idx   = None
    result     = None

    # 模拟持有直到止盈/止损
    for offset in range(1, len(df) - entry_idx):
        h = float(high_prices[entry_idx + offset])
        l = float(low_prices[entry_idx + offset])
        peak = max(peak, h)
        trail_stop = peak * (1 - TRAILING_PCT)
        fixed_stop = entry_price * (1 - STOP_PCT)
        stop_level = max(trail_stop, fixed_stop)
        if l <= stop_level:
            result     = 'win' if peak > entry_price else 'loss'
            exit_price = stop_level
            exit_idx   = entry_idx + offset
            break

    # 未触发止盈止损，标记为“未平仓”
    if result is None:
        open_trades.append({
            'entry_time': entry_time,
            'entry':      entry_price
        })
    else:
        closed_trades.append({
            'entry_time': entry_time,
            'entry':      entry_price,
            'exit_time':  df.index[exit_idx],
            'exit':       exit_price,
            'result':     result
        })

# ====== 构造 Telegram 消息 ======
msg = []

# 1. 历史回测部分（Closed）
if closed_trades:
    df_closed = pd.DataFrame(closed_trades)
    df_closed['profit_pct'] = (df_closed['exit'] - df_closed['entry']) / df_closed['entry'] * 100
    # 把 DataFrame 转成 Markdown 表格
    table_md = df_closed.rename(columns={
        'entry_time':'EntryTime','entry':'Entry','exit_time':'ExitTime','exit':'Exit','result':'Result','profit_pct':'PnL(%)'
    }).to_markdown(index=False)
    cap = INITIAL_CAPITAL
    for pct in df_closed['profit_pct']:
        cap *= (1 + pct/100)
    cum_ret = (cap/INITIAL_CAPITAL - 1)*100
    msg.append("=== 历史回测结果 ===")
    msg.append(table_md)
    msg.append(f"累计回报：{cum_ret:.2f}% (初始 {INITIAL_CAPITAL:.2f} → 最终 {cap:.2f})\n")
else:
    msg.append("⚠️ 无历史回测信号\n")

# 2. 当前未平仓交易（Open）
if open_trades:
    current_price = float(df['Close'].iloc[-1])
    msg.append("=== 未平仓交易 ===")
    for ot in open_trades:
        unreal_pnl = (current_price - ot['entry'])/ot['entry']*100
        msg.append(
            f"Entry: {ot['entry_time'].strftime('%Y-%m-%d %H:%M')} @ {ot['entry']:.2f}  "
            f"最新价：{current_price:.2f}  Unrealized: {unreal_pnl:.2f}%"
        )
else:
    msg.append("📊 今日暂无未平仓信号")

final_msg = "\n".join(msg)

# 3. 发送文字报告
bot.send_message(chat_id=CHAT_ID, text=final_msg, parse_mode='Markdown')

# ====== 绘图（仅本地调试用，如需上传到 Telegram 可解注释） =====#
if pattern_points:
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df['Close'], color='gray', alpha=0.5, label='Close')
    plotted = set()
    def safe_label(lbl):
        if lbl in plotted: return "_nolegend_"
        plotted.add(lbl); return lbl

    # 标注已平仓进出场
    for tr in closed_trades:
        ax.scatter(tr['entry_time'], tr['entry'], marker='^', c='green', label=safe_label('Entry'))
        ax.scatter(tr['exit_time'],  tr['exit'],  marker='v', c='red',   label=safe_label('Exit'))
    # 标注未平仓进场
    for ot in open_trades:
        ax.scatter(ot['entry_time'], ot['entry'], marker='*', s=150, c='orange', label=safe_label('Open'))

    ax.set_title(f"{TICKER} W-Pattern Strategy")
    ax.set_xlabel("Time"); ax.set_ylabel("Price")
    ax.legend(loc="best"); ax.grid(True); plt.tight_layout()
    # plt.savefig("w_pattern_plot.png")
    # plt.show()
