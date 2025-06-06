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

# 初始化 Telegram Bot（注意：python-telegram-bot v20 以后使用异步接口）
bot = Bot(token=BOT_TOKEN)

# ====== 参数区（方便调整） ======
TICKER           = "2330.TW"
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
TRAILING_PCT     = 0.08
STOP_PCT         = 0.10

# ====== 数据下载 ======
# 注意：yfinance.download() 中 auto_adjust 的默认值已变为 True，如需关闭请显式传 auto_adjust=False
df = yf.download(TICKER, interval=INTERVAL, period=PERIOD, auto_adjust=False)
df.dropna(inplace=True)

# 将价格列转为 numpy array，便于快速访问
close_prices = df["Close"].to_numpy()
high_prices  = df["High"].to_numpy()
low_prices   = df["Low"].to_numpy()

# ====== 寻找 W 底信号 ======
pullback_signals = []   # 存放 (entry_idx, entry_price, neckline) 的列表
pattern_points   = []   # 如果以后想画出形态结构，可用到

def detect_w(min_idx, max_idx, tol_p1p3, lo, hi):
    """
    在 min_idx、max_idx 指定的极值点之间寻找符合 W 底的信号：
    - tol_p1p3: P1 与 P3 之间价格相似度容差
    - lo/hi：颈线拉回允许的下限/上限
    检测到的结果 append 到 pullback_signals，结构为 (trigger_idx, trigger_price, neckline_price)
    """
    for i in range(1, len(min_idx)):
        p1 = int(min_idx[i-1])
        p3 = int(min_idx[i])

        # p2 必须是 p1 ~ p3 之间的最高点
        mids = max_idx[(max_idx > p1) & (max_idx < p3)]
        if mids.size == 0:
            continue
        p2 = int(mids[-1])

        # 取出收盘价作为 float
        p1v = float(close_prices[p1].item())
        p2v = float(close_prices[p2].item())
        p3v = float(close_prices[p3].item())

        # 基本形态：两头低，中间高
        if not (p1v < p2v and p3v < p2v):
            continue

        # P1 与 P3 必须在 tol_p1p3 范围内
        if abs(p1v - p3v) / p1v > tol_p1p3:
            continue

        neckline = p2v
        bo_i     = p3 + 1
        # 如果越界，跳过
        if bo_i + 4 >= len(close_prices):
            continue

        bo_v = float(close_prices[bo_i].item())      # 突破点价格
        pb_v = float(close_prices[bo_i + 2].item())  # 拉回点价格
        tr_v = float(close_prices[bo_i + 4].item())  # 触发点价格

        # 1) 突破必须高于 颈线 * (1 + BREAKOUT_PCT)
        if bo_v <= neckline * (1 + BREAKOUT_PCT):
            continue

        # 2) 拉回点必须在 [neckline * lo, neckline * hi] 区间
        if not (neckline * lo < pb_v < neckline * hi):
            continue

        # 3) 触发点必须高于拉回点
        if tr_v <= pb_v:
            continue

        # 符合「W 底回测进场」条件，记录这个信号
        pullback_signals.append((bo_i + 4, tr_v, neckline))
        pattern_points.append((p1, p1v, p2, p2v, p3, p3v, bo_i, bo_v, pb_v, tr_v, tol_p1p3))


# -------- 小型 W --------
min_idx_small = argrelextrema(close_prices, np.less_equal,    order=MIN_ORDER_SMALL)[0]
max_idx_small = argrelextrema(close_prices, np.greater_equal, order=MIN_ORDER_SMALL)[0]
detect_w(min_idx_small, max_idx_small,
         P1P3_TOL_SMALL, PULLBACK_LO_SMALL, PULLBACK_HI_SMALL)

# -------- 大型 W --------
min_idx_large = argrelextrema(close_prices, np.less_equal,    order=MIN_ORDER_LARGE)[0]
max_idx_large = argrelextrema(close_prices, np.greater_equal, order=MIN_ORDER_LARGE)[0]
detect_w(min_idx_large, max_idx_large,
         P1P3_TOL_LARGE, PULLBACK_LO_LARGE, PULLBACK_HI_LARGE)

# ====== 回测阶段：分别区分「已平仓」与「未平仓」 ======
completed_trades = []  # 每笔已平仓交易：{'entry_time','entry','exit_time','exit'}
open_trades      = []  # 每笔未平仓交易：{'entry_time','entry'}

for entry_idx, entry_price, neckline in pullback_signals:
    entry_time  = df.index[entry_idx]
    peak        = entry_price
    exit_price  = None
    exit_idx    = None
    result_flag = False  # True 表示触发了止盈/止损

    # 持有阶段：从 entry_idx+1 往后找，直到触发止盈/止损或到数据末尾
    for offset in range(1, len(df) - entry_idx):
        h = float(high_prices[entry_idx + offset].item())
        l = float(low_prices[entry_idx + offset].item())
        peak = max(peak, h)

        # 计算移动止盈和固定止损
        trail_stop = peak * (1 - TRAILING_PCT)
        fixed_stop = entry_price * (1 - STOP_PCT)
        stop_level = max(trail_stop, fixed_stop)

        # 如果当天最低价 <= 止损线，就在此退出
        if l <= stop_level:
            exit_price  = stop_level
            exit_idx    = entry_idx + offset
            result_flag = True
            break

    if result_flag:
        # 已平仓交易
        exit_time = df.index[exit_idx]
        completed_trades.append({
            'entry_time': entry_time,
            'entry':      entry_price,
            'exit_time':  exit_time,
            'exit':       exit_price
        })
    else:
        # 到最后一天都没触发止损/止盈，视为「未平仓」
        open_trades.append({
            'entry_time': entry_time,
            'entry':      entry_price
        })

# ====== 构造要发送到 Telegram 的文本消息 ======
msg_lines = []
msg_lines.append(f"📊 历史回测： 共 {len(completed_trades)} 笔已完成交易")
cap = INITIAL_CAPITAL

for idx, tr in enumerate(completed_trades, start=1):
    e_price = float(tr["entry"])
    x_price = float(tr["exit"])
    profit_pct = (x_price - e_price) / e_price * 100
    cap *= (1 + profit_pct / 100)

    msg_lines.append(
        f"{idx}. Entry: {tr['entry_time'].strftime('%Y-%m-%d %H:%M')} @ {e_price:.2f}    "
        f"Exit:  {tr['exit_time'].strftime('%Y-%m-%d %H:%M')} @ {x_price:.2f}    "
        f"Profit: {profit_pct:.2f}%"
    )

cum_ret = (cap / INITIAL_CAPITAL - 1) * 100
msg_lines.append(f"\n初始资金：{INITIAL_CAPITAL:.2f} → 最终资金：{cap:.2f} ，累计回报：{cum_ret:.2f}%")

if open_trades:
    msg_lines.append(f"\n📌 当前共有 {len(open_trades)} 笔未平仓：")
    for idx, ot in enumerate(open_trades, start=1):
        msg_lines.append(
            f"{idx}. Entry: {ot['entry_time'].strftime('%Y-%m-%d %H:%M')} @ {ot['entry']:.2f}"
        )

final_msg = "\n".join(msg_lines)

# —— 异步发送文字消息 —— #
async def _send_text():
    await bot.send_message(chat_id=CHAT_ID, text=final_msg)

asyncio.run(_send_text())


# ====== 绘图并上传到 Telegram =====#
# “只保留已平仓的进/出场点 (绿/红) 以及未平仓的进场点 (黄)”
if completed_trades or open_trades:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["Close"], color="gray", alpha=0.5, label="Close")
    plotted = set()

    def safe_label(lbl):
        if lbl in plotted:
            return "_nolegend_"
        plotted.add(lbl)
        return lbl

    # 标注“已平仓”交易的进/出场
    for tr in completed_trades:
        ax.scatter(
            tr["entry_time"], tr["entry"],
            marker="^", c="green", s=50,
            label=safe_label("Entry")
        )
        ax.scatter(
            tr["exit_time"], tr["exit"],
            marker="v", c="red", s=50,
            label=safe_label("Exit")
        )

    # 标注“未平仓”交易的进场点
    for ot in open_trades:
        ax.scatter(
            ot["entry_time"], ot["entry"],
            marker="^", c="yellow", edgecolors="black", s=80,
            label=safe_label("Open Entry")
        )

    ax.set_title(f"{TICKER} W-Pattern Strategy")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend(loc="best")
    ax.grid(True)
    plt.tight_layout()

    chart_file = "w_pattern_plot.png"
    plt.savefig(chart_file)
    plt.close()

    # —— 异步发送图片 —— #
    async def _send_photo():
        with open(chart_file, "rb") as img:
            await bot.send_photo(chat_id=CHAT_ID, photo=img)

    asyncio.run(_send_photo())
