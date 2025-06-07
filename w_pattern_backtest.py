#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
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

bot = Bot(token=BOT_TOKEN)

# ====== 参数区 ======
TICKER = "2330.tw"
INTERVAL = "60m"
PERIOD   = "600d"

MIN_ORDER_SMALL   = 3
P1P3_TOL_SMALL    = 0.9
PULLBACK_LO_SMALL = 0.8
PULLBACK_HI_SMALL = 1.2

MIN_ORDER_LARGE   = 200
P1P3_TOL_LARGE    = 0.9
PULLBACK_LO_LARGE = 0.78
PULLBACK_HI_LARGE = 1.4

BREAKOUT_PCT    = 0.00001
INITIAL_CAPITAL = 100.0
TRAILING_PCT    = 0.08
STOP_PCT        = 0.10

# ====== 下载数据 ======
df = yf.download(TICKER, interval=INTERVAL, period=PERIOD, auto_adjust=False)
df.dropna(inplace=True)
close_prices = df['Close'].to_numpy()
high_prices  = df['High'].to_numpy()
low_prices   = df['Low'].to_numpy()

# ====== W 底检测 ======
pullback_signals = []   # (entry_idx, entry_price, neckline)
pattern_points   = []

def detect_w(min_idx, max_idx, tol, lo, hi):
    for i in range(1, len(min_idx)):
        p1 = int(min_idx[i-1]); p3 = int(min_idx[i])
        mids = max_idx[(max_idx>p1)&(max_idx<p3)]
        if mids.size==0: continue
        p2 = int(mids[-1])
        p1v = close_prices[p1].item()
        p2v = close_prices[p2].item()
        p3v = close_prices[p3].item()
        if not (p1v<p2v and p3v<p2v): continue
        if abs(p1v-p3v)/p1v>tol: continue
        neckline = p2v; bo_i = p3+1
        if bo_i+4>=len(close_prices): continue
        bo_v = close_prices[bo_i].item()
        pb_v = close_prices[bo_i+2].item()
        tr_v = close_prices[bo_i+4].item()
        if bo_v<=neckline*(1+BREAKOUT_PCT): continue
        if not (neckline*lo<pb_v<neckline*hi): continue
        if tr_v<=pb_v: continue

        pullback_signals.append((bo_i+4, tr_v, neckline))
        pattern_points.append((p1,p1v,p2,p2v,p3,p3v,bo_i,bo_v,pb_v,tr_v,tol))

# 小型
mi_s = argrelextrema(close_prices, np.less_equal, order=MIN_ORDER_SMALL)[0]
ma_s = argrelextrema(close_prices, np.greater_equal, order=MIN_ORDER_SMALL)[0]
detect_w(mi_s, ma_s, P1P3_TOL_SMALL, PULLBACK_LO_SMALL, PULLBACK_HI_SMALL)
# 大型
mi_l = argrelextrema(close_prices, np.less_equal, order=MIN_ORDER_LARGE)[0]
ma_l = argrelextrema(close_prices, np.greater_equal, order=MIN_ORDER_LARGE)[0]
detect_w(mi_l, ma_l, P1P3_TOL_LARGE, PULLBACK_LO_LARGE, PULLBACK_HI_LARGE)

# ====== 回测 & 区分平仓/未平仓 ======
closed, open_trades = [], []
for entry_idx, entry_price, neckline in pullback_signals:
    entry_time = df.index[entry_idx]
    peak       = entry_price
    exit_price = None; exit_idx=None; result=None
    for off in range(1, len(df)-entry_idx):
        h = high_prices[entry_idx+off].item()
        l = low_prices[entry_idx+off].item()
        peak = max(peak, h)
        trail_stop = peak*(1-TRAILING_PCT)
        fixed_stop = entry_price*(1-STOP_PCT)
        stop_lv = max(trail_stop, fixed_stop)
        if l<=stop_lv:
            result='win' if peak>entry_price else 'loss'
            exit_price=stop_lv; exit_idx=entry_idx+off
            break
    if result is None:
        open_trades.append({'entry_time':entry_time,'entry':entry_price})
    else:
        closed.append({
            'entry_time': entry_time,
            'entry':      entry_price,
            'exit_time':  df.index[exit_idx],
            'exit':       exit_price,
            'result':     result
        })

# ====== 构造消息 ======
lines = []

# 历史平仓
if closed:
    dfc = pd.DataFrame(closed)
    dfc['pnl'] = (dfc['exit']-dfc['entry'])/dfc['entry']*100
    # 手动拼 Markdown 表格
    hdr = ["EntryTime","Entry","ExitTime","Exit","Result","PnL(%)"]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "|".join(["---"]*len(hdr)) + "|")
    for _,r in dfc.iterrows():
        lines.append(
            "| " +
            f"{r['entry_time'].strftime('%Y-%m-%d %H:%M')} | "
            f"{r['entry']:.2f} | "
            f"{r['exit_time'].strftime('%Y-%m-%d %H:%M')} | "
            f"{r['exit']:.2f} | "
            f"{r['result']} | "
            f"{r['pnl']:.2f} |"
        )
    cap = INITIAL_CAPITAL
    for pct in dfc['pnl']: cap *= (1+ pct/100)
    cum = (cap/INITIAL_CAPITAL-1)*100
    lines.append(f"\n累计回报：{cum:.2f}% (初 {INITIAL_CAPITAL:.2f}→末 {cap:.2f})\n")
else:
    lines.append("⚠️ 无历史平仓信号\n")

# 未平仓
if open_trades:
    curp = close_prices[-1].item()
    lines.append("=== 未平仓交易 ===")
    for ot in open_trades:
        unreal = (curp-ot['entry'])/ot['entry']*100
        lines.append(
            f"Entry: {ot['entry_time'].strftime('%Y-%m-%d %H:%M')} @ {ot['entry']:.2f}  "
            f"现价: {curp:.2f}  Unreal: {unreal:.2f}%"
        )
else:
    lines.append("📊 今日无未平仓信号")

final_msg = "\n".join(lines)
bot.send_message(chat_id=CHAT_ID, text=final_msg, parse_mode='Markdown')

# ====== 绘图（本地调试可开启） =====#
if pattern_points:
    fig,ax=plt.subplots(figsize=(12,6))
    ax.plot(df['Close'],color='gray',alpha=0.5,label='Close')
    seen=set()
    def lbl(x):
        if x in seen: return "_nolegend_"
        seen.add(x); return x

    for tr in closed:
        ax.scatter(tr['entry_time'],tr['entry'],marker='^',c='green',label=lbl('Entry'))
        ax.scatter(tr['exit_time'],tr['exit'],marker='v',c='red',label=lbl('Exit'))
    for ot in open_trades:
        ax.scatter(ot['entry_time'],ot['entry'],marker='*',s=150,c='orange',label=lbl('Open'))

    ax.set_title(f"{TICKER} W-Pattern")
    ax.set_xlabel("Time"); ax.set_ylabel("Price")
    ax.legend(loc="best"); ax.grid(True); plt.tight_layout()
    # plt.savefig("w_pattern_plot.png")
