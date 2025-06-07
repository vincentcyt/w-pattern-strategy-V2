#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from telegram import Bot
from io import BytesIO

# —— 环境变量检查 —— #
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")
if not BOT_TOKEN or not CHAT_ID:
    print("❌ ERROR: 环境变量 BOT_TOKEN 或 CHAT_ID 未设置，程序退出。")
    sys.exit(1)

bot = Bot(token=BOT_TOKEN)

# ====== 参数区 ======
TICKER = "2330.tw"
INTERVAL = "60m"
PERIOD = "600d"

# 小型 W 参数
MIN_ORDER_SMALL   = 3
P1P3_TOL_SMALL    = 0.15
PULLBACK_LO_SMALL = 0.99
PULLBACK_HI_SMALL = 1.01

# 大型 W 参数
MIN_ORDER_LARGE   = 200
P1P3_TOL_LARGE    = 0.25
PULLBACK_LO_LARGE = 0.95
PULLBACK_HI_LARGE = 1.05

# 回测参数
BREAKOUT_PCT    = 0.005
INITIAL_CAPITAL = 100.0
TRAILING_PCT    = 0.07
STOP_PCT        = 0.03

# ====== 拉取数据 ======
df = yf.download(
    TICKER,
    interval=INTERVAL,
    period=PERIOD,
    auto_adjust=False  # 如果你想要复权价请设为 True
)
df.dropna(inplace=True)

close_prices = df['Close'].to_numpy()
high_prices  = df['High'].to_numpy()
low_prices   = df['Low'].to_numpy()

# ====== W 底检测 ======
pullback_signals = []   # (entry_idx, entry_price, neckline)
pattern_points   = []   # 用于画图

def detect_w(min_idx, max_idx, tol, lo, hi):
    for i in range(1, len(min_idx)):
        p1 = int(min_idx[i-1])
        p3 = int(min_idx[i])
        mids = max_idx[(max_idx > p1) & (max_idx < p3)]
        if mids.size == 0: continue
        p2 = int(mids[-1])

        p1v, p2v, p3v = map(lambda x: float(close_prices[x].item()), (p1, p2, p3))
        if not (p1v < p2v and p3v < p2v): continue
        if abs(p1v - p3v)/p1v > tol: continue

        neckline = p2v
        bo_i     = p3 + 1
        if bo_i + 4 >= len(close_prices): continue
        bo_v = float(close_prices[bo_i].item())
        pb_v = float(close_prices[bo_i+2].item())
        tr_v = float(close_prices[bo_i+4].item())

        if bo_v <= neckline*(1+BREAKOUT_PCT): continue
        if not (neckline*lo < pb_v < neckline*hi): continue
        if tr_v <= pb_v: continue

        pullback_signals.append((bo_i+4, tr_v, neckline))
        pattern_points.append((p1,p1v,p2,p2v,p3,p3v,bo_i,bo_v,pb_v,tr_v,tol))

# 小型W
min_s = argrelextrema(close_prices, np.less_equal, order=MIN_ORDER_SMALL)[0]
max_s = argrelextrema(close_prices, np.greater_equal, order=MIN_ORDER_SMALL)[0]
detect_w(min_s, max_s, P1P3_TOL_SMALL, PULLBACK_LO_SMALL, PULLBACK_HI_SMALL)
# 大型W
min_L = argrelextrema(close_prices, np.less_equal, order=MIN_ORDER_LARGE)[0]
max_L = argrelextrema(close_prices, np.greater_equal, order=MIN_ORDER_LARGE)[0]
detect_w(min_L, max_L, P1P3_TOL_LARGE, PULLBACK_LO_LARGE, PULLBACK_HI_LARGE)

# ====== 回测 ======
results = []
for entry_idx, entry_price, neckline in pullback_signals:
    entry_time = df.index[entry_idx]
    peak = entry_price
    result = None
    exit_price = None
    exit_idx = None

    for offset in range(1, len(df)-entry_idx):
        h = float(high_prices[entry_idx+offset].item())
        l = float(low_prices[entry_idx+offset].item())
        peak = max(peak, h)

        trail_stop = peak*(1-TRAILING_PCT)
        fixed_stop = entry_price*(1-STOP_PCT)
        stop_lvl   = max(trail_stop, fixed_stop)

        if l <= stop_lvl:
            result     = 'win' if peak>entry_price else 'loss'
            exit_price = stop_lvl
            exit_idx   = entry_idx+offset
            break

    if result is None:
        exit_idx   = len(df)-1
        exit_price = float(close_prices[exit_idx].item())
        result     = 'win' if exit_price>entry_price else 'loss'

    results.append({
        'entry_time': entry_time,
        'entry':      entry_price,
        'exit_time':  df.index[exit_idx],
        'exit':       exit_price,
        'result':     result
    })

# 历史回测表
results_df = pd.DataFrame(results)
if not results_df.empty:
    results_df['profit_pct'] = (results_df['exit']-results_df['entry'])/results_df['entry']*100
    total_trades = len(results_df)
    cap = INITIAL_CAPITAL
    for pct in results_df['profit_pct']: cap *= (1+float(pct)/100)
    cum_ret = (cap/INITIAL_CAPITAL-1)*100
else:
    total_trades = 0
    cap = INITIAL_CAPITAL
    cum_ret = 0.0

# 当日信号（UTC→Local 可自行调整）
today = pd.Timestamp.utcnow().normalize()
today_signals = results_df[
    results_df['entry_time'].dt.tz_convert('UTC').dt.normalize() == today
] if not results_df.empty else pd.DataFrame()

# 构建 Markdown 消息
lines = []
if not today_signals.empty:
    lines.append("✅ *今日新 W 底信号*")
    lines.append("|#|Entry Time|Entry|Exit Time|Exit|Profit|")
    lines.append("|-:|:--|--:|:--|--:|--:|")
    for i,row in today_signals.iterrows():
        lines.append(
            f"|{i+1}|{row['entry_time'].strftime('%Y-%m-%d %H:%M')}|{row['entry']:.2f}|"
            f"{row['exit_time'].strftime('%Y-%m-%d %H:%M')}|{row['exit']:.2f}|{row['profit_pct']:.2f}%|"
        )
else:
    lines.append("📊 *今日无 W 底信号*")

lines.append("\n✅ *历史回测汇总*")
lines.append(f"- 总交易笔数：{total_trades}")
lines.append(f"- 初始资金：{INITIAL_CAPITAL:.2f} → 最终资金：{cap:.2f}")
lines.append(f"- 累计回报：{cum_ret:.2f}%")

final_msg = "\n".join(lines)

# 画图（只保留进/出点 + 未平仓进场点）
fig,ax = plt.subplots(figsize=(12,5))
ax.plot(df['Close'], color='lightgray', label='Close')
plotted=set()
def sl(lbl):
    if lbl in plotted: return "_nolegend_"
    plotted.add(lbl)
    return lbl

# 全部已平仓点
for _,r in results_df.iterrows():
    ax.scatter(r['entry_time'], r['entry'], marker='^', c='green', label=sl('Entry'))
    ax.scatter(r['exit_time'],  r['exit'],  marker='v', c='red',   label=sl('Exit'))

# 未平仓进场
open_trades = today_signals[today_signals['exit_time']==today_signals['entry_time']]
for _,r in open_trades.iterrows():
    ax.scatter(r['entry_time'], r['entry'], marker='^', c='blue', label=sl('Open'))

ax.set_title(f"{TICKER} W-Pattern 回测")
ax.set_xlabel("Time"); ax.set_ylabel("Price")
ax.legend(loc='best'); ax.grid(True); plt.tight_layout()

# 把图存到内存
buf = BytesIO()
fig.savefig(buf, format='png')
buf.seek(0)

# —— 异步发送消息 + 图片 —— #
async def main():
    await bot.send_message(chat_id=CHAT_ID, text=final_msg, parse_mode='Markdown')
    await bot.send_photo(chat_id=CHAT_ID, photo=buf)

if __name__=="__main__":
    asyncio.run(main())
