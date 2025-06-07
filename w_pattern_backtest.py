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

# —— 调试：打印环境变量是否存在 —— #
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")
print(f"[DEBUG] BOT_TOKEN is [{'set' if BOT_TOKEN else 'NOT set'}]")
print(f"[DEBUG] CHAT_ID   is [{'set' if CHAT_ID else 'NOT set'}]")

if not BOT_TOKEN or not CHAT_ID:
    print("❌ ERROR: 环境变量 BOT_TOKEN 或 CHAT_ID 不存在，程序退出。")
    sys.exit(1)

# 初始化 Telegram Bot（python-telegram-bot v20+ 使用异步接口）
bot = Bot(token=BOT_TOKEN)

# ====== 参数区（方便调整） ======
TICKER           = "2330.TW"   # 注意：yfinance 要求大写后缀
INTERVAL         = "60m"
PERIOD           = "600d"

# 小型 W 参数
MIN_ORDER_SMALL  = 3
P1P3_TOL_SMALL   = 0.9
PULLBACK_LO_SMALL, PULLBACK_HI_SMALL = 0.8, 1.2

# 大型 W 参数
MIN_ORDER_LARGE  = 200
P1P3_TOL_LARGE   = 0.9
PULLBACK_LO_LARGE, PULLBACK_HI_LARGE = 0.78, 1.4

# 统一参数
BREAKOUT_PCT     = 0.00001
INITIAL_CAPITAL  = 100.0
TRAILING_PCT     = 0.07
STOP_PCT         = 0.03

# ====== 数据下载 ======
# 注意：yfinance.download() 中 auto_adjust 的默认值已改为 True，如果想关闭请显式传 auto_adjust=False
df = yf.download(TICKER, interval=INTERVAL, period=PERIOD, auto_adjust=False)
df.dropna(inplace=True)

# 转为 numpy arrays，便于快速索引
close_prices = df["Close"].to_numpy()
high_prices  = df["High"].to_numpy()
low_prices   = df["Low"].to_numpy()

# ====== 寻找 W 底信号 ======
pullback_signals = []   # 存放 (trigger_idx, trigger_price, neckline_price)
pattern_points   = []   # 如果以后需要画结构，可以保留这些点

def detect_w(min_idx, max_idx, tol_p1p3, lo, hi):
    for i in range(1, len(min_idx)):
        p1 = int(min_idx[i-1])
        p3 = int(min_idx[i])
        mids = max_idx[(max_idx > p1) & (max_idx < p3)]
        if mids.size == 0: continue
        p2 = int(mids[-1])

        p1v = float(close_prices[p1].item())
        p2v = float(close_prices[p2].item())
        p3v = float(close_prices[p3].item())
        if not (p1v < p2v and p3v < p2v): continue
        if abs(p1v - p3v)/p1v > tol_p1p3: continue

        neckline = p2v
        bo_i     = p3 + 1
        if bo_i+4 >= len(close_prices): continue

        bo_v = float(close_prices[bo_i].item())
        pb_v = float(close_prices[bo_i+2].item())
        tr_v = float(close_prices[bo_i+4].item())
        if bo_v <= neckline*(1+BREAKOUT_PCT): continue
        if not (neckline*lo < pb_v < neckline*hi): continue
        if tr_v <= pb_v: continue

        pullback_signals.append((bo_i+4, tr_v, neckline))
        pattern_points.append((p1,p1v,p2,p2v,p3,p3v,bo_i,bo_v,pb_v,tr_v,tol_p1p3))

# 小型 W
min_s = argrelextrema(close_prices, np.less_equal,    order=MIN_ORDER_SMALL)[0]
max_s = argrelextrema(close_prices, np.greater_equal, order=MIN_ORDER_SMALL)[0]
detect_w(min_s, max_s, P1P3_TOL_SMALL, PULLBACK_LO_SMALL, PULLBACK_HI_SMALL)

# 大型 W
min_L = argrelextrema(close_prices, np.less_equal,    order=MIN_ORDER_LARGE)[0]
max_L = argrelextrema(close_prices, np.greater_equal, order=MIN_ORDER_LARGE)[0]
detect_w(min_L, max_L, P1P3_TOL_LARGE, PULLBACK_LO_LARGE, PULLBACK_HI_LARGE)

# ====== 回测阶段：分别区分“已平仓”与“未平仓” ======
completed_trades = []  # 已平仓
open_trades      = []  # 未平仓

for entry_idx, entry_price, neckline in pullback_signals:
    entry_time = df.index[entry_idx]
    peak       = entry_price
    exit_price = None
    exit_idx   = None
    triggered  = False

    for offset in range(1, len(df)-entry_idx):
        h = float(high_prices[entry_idx+offset].item())
        l = float(low_prices[entry_idx+offset].item())
        peak = max(peak, h)
        trail_stop = peak*(1-TRAILING_PCT)
        fixed_stop = entry_price*(1-STOP_PCT)
        stop_lvl   = max(trail_stop, fixed_stop)
        if l <= stop_lvl:
            exit_price = stop_lvl
            exit_idx   = entry_idx+offset
            triggered  = True
            break

    if triggered:
        completed_trades.append({
            "entry_time": entry_time,
            "entry":      entry_price,
            "exit_time":  df.index[exit_idx],
            "exit":       exit_price
        })
    else:
        # 未平仓，记录 entry
        open_trades.append({
            "entry_time": entry_time,
            "entry":      entry_price
        })

# ====== 判断“今日是否有交易信号” ======
today = pd.Timestamp.utcnow().date()
has_signal_today = False
for t in completed_trades:
    if t["entry_time"].tz_convert("UTC").date() == today:
        has_signal_today = True
        break
if not has_signal_today:
    for ot in open_trades:
        if ot["entry_time"].tz_convert("UTC").date() == today:
            has_signal_today = True
            break

# ====== 构造 Telegram 文本消息 ======
# 1) 历史已平仓表格
if completed_trades:
    comp_df = pd.DataFrame(completed_trades)
    comp_df["profit_pct"] = (comp_df["exit"] - comp_df["entry"]) / comp_df["entry"] * 100
    cap = INITIAL_CAPITAL
    for pct in comp_df["profit_pct"]:
        cap *= (1 + float(pct)/100)
    cum_ret = (cap/INITIAL_CAPITAL - 1)*100

    table_txt = comp_df.to_string(
        index=False,
        columns=["entry_time","entry","exit_time","exit","profit_pct"],
        justify="left",
        formatters={
            "entry_time": lambda v: v.strftime("%Y-%m-%d %H:%M"),
            "exit_time":  lambda v: v.strftime("%Y-%m-%d %H:%M"),
            "entry":      lambda v: f"{v:.2f}",
            "exit":       lambda v: f"{v:.2f}",
            "profit_pct": lambda v: f"{v:.2f}%"
        }
    )

    header = f"📊 历史回测（共 {len(completed_trades)} 笔）"
    summary = f"初始资金：{INITIAL_CAPITAL:.2f} → 最终资金：{cap:.2f} ，累计回报：{cum_ret:.2f}%"
else:
    header = "📊 历史回测：无已平仓交易"
    summary = f"初始资金：{INITIAL_CAPITAL:.2f} → {INITIAL_CAPITAL:.2f} ，累计回报：0.00%"
    table_txt = ""

# 2) 当日信号
today_line = f"📅 今日是否有交易信號：{'✅ 有' if has_signal_today else '❌ 無'}"

# 3) 未平仓交易：加上最新价格与未实现盈亏
open_txt = ""
if open_trades:
    latest_price = float(df["Close"].iloc[-1])
    open_lines = [f"📌 当前未平倉（共 {len(open_trades)} 笔）："]
    for idx, ot in enumerate(open_trades, 1):
        pnl_pct = (latest_price - ot["entry"])/ot["entry"]*100
        open_lines.append(
            f"{idx}. Entry: {ot['entry_time'].strftime('%Y-%m-%d %H:%M')} @ {ot['entry']:.2f}  "
            f"現價: {latest_price:.2f}  未實盈虧: {pnl_pct:.2f}%"
        )
    open_txt = "\n" + "\n".join(open_lines)

# 汇总
parts = [today_line, header]
if table_txt:
    parts.append(f"```\n{table_txt}\n```")
parts.append(summary)
parts.append(open_txt)
final_msg = "\n".join(parts)

# ====== 画图：只标注已平仓进/出点 & 未平仓进场点 ======
chart_file = "w_pattern_plot.png"
if completed_trades or open_trades:
    fig,ax = plt.subplots(figsize=(12,6))
    ax.plot(df["Close"], color="lightgray", label="Close")
    plotted = set()
    def sl(lbl):
        if lbl in plotted: return "_nolegend_"
        plotted.add(lbl)
        return lbl

    # 已平仓
    for tr in completed_trades:
        ax.scatter(tr["entry_time"], tr["entry"], marker="^", c="green", s=50, label=sl("Entry"))
        ax.scatter(tr["exit_time"],  tr["exit"],  marker="v", c="red",   s=50, label=sl("Exit"))
    # 未平仓
    for ot in open_trades:
        ax.scatter(ot["entry_time"], ot["entry"], marker="^", c="orange", s=80, edgecolors="black", label=sl("Open"))

    ax.set_title(f"{TICKER} W-Pattern Strategy")
    ax.set_xlabel("Time"); ax.set_ylabel("Price")
    ax.legend(loc="best"); ax.grid(True); plt.tight_layout()
    plt.savefig(chart_file); plt.close()

# ====== 异步发送到 Telegram ======
async def main():
    # 文字
    await bot.send_message(chat_id=CHAT_ID, text=final_msg, parse_mode="Markdown")
    # 图片
    if os.path.exists(chart_file):
        with open(chart_file,"rb") as img:
            await bot.send_photo(chat_id=CHAT_ID, photo=img)

if __name__=="__main__":
    asyncio.run(main())
