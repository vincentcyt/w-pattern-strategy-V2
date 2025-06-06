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

# 初始化 Telegram Bot（python-telegram-bot v20+ 中 send_message/send_photo 都是 coroutine）
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
# “auto_adjust=False” 保持原始 OHLC
df = yf.download(TICKER, interval=INTERVAL, period=PERIOD, auto_adjust=False)
df.dropna(inplace=True)

# 转成 numpy arrays
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
        p1 = int(min_idx[i - 1])
        p3 = int(min_idx[i])

        # p2 必须是 p1~p3 之间的最高点
        mids = max_idx[(max_idx > p1) & (max_idx < p3)]
        if mids.size == 0:
            continue
        p2 = int(mids[-1])

        # 取出收盘价（float）
        p1v = float(close_prices[p1].item())
        p2v = float(close_prices[p2].item())
        p3v = float(close_prices[p3].item())

        # 两头低中间高
        if not (p1v < p2v and p3v < p2v):
            continue

        # P1 与 P3 要够相近
        if abs(p1v - p3v) / p1v > tol_p1p3:
            continue

        neckline = p2v
        bo_i     = p3 + 1
        if bo_i + 4 >= len(close_prices):
            continue

        bo_v = float(close_prices[bo_i].item())
        pb_v = float(close_prices[bo_i + 2].item())
        tr_v = float(close_prices[bo_i + 4].item())

        # 突破点要比颈线高
        if bo_v <= neckline * (1 + BREAKOUT_PCT):
            continue

        # 拉回要在可接受区间
        if not (neckline * lo < pb_v < neckline * hi):
            continue

        # 触发点要高于拉回点
        if tr_v <= pb_v:
            continue

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
completed_trades = []  # 已触及止盈/止损的
open_trades      = []  # 尚未触及的“开仓但未平仓”

for entry_idx, entry_price, neckline in pullback_signals:
    entry_time = df.index[entry_idx]
    peak       = entry_price
    result     = None
    exit_price = None
    exit_idx   = None

    # 从 entry_idx+1 向后遍历，看是否碰到止损/止盈
    for offset in range(1, len(df) - entry_idx):
        h = float(high_prices[entry_idx + offset].item())
        l = float(low_prices[entry_idx + offset].item())
        peak = max(peak, h)

        trail_stop = peak * (1 - TRAILING_PCT)
        fixed_stop = entry_price * (1 - STOP_PCT)
        stop_level = max(trail_stop, fixed_stop)

        if l <= stop_level:
            # 已触及止盈/止损
            result     = 'win' if peak > entry_price else 'loss'
            exit_price = stop_level
            exit_idx   = entry_idx + offset
            break

    if result is not None:
        completed_trades.append({
            'entry_time': entry_time,
            'entry':      entry_price,
            'exit_time':  df.index[exit_idx],
            'exit':       exit_price,
            'result':     result
        })
    else:
        # 数据结尾还没触及
        open_trades.append({
            'entry_time': entry_time,
            'entry':      entry_price,
            'reason':     '尚未触及止盈/止损'
        })

# ====== 已完成交易整理 ======
if completed_trades:
    results_df = pd.DataFrame(completed_trades)
    results_df['profit_pct'] = (results_df['exit'] - results_df['entry']) / results_df['entry'] * 100

    # 构造等宽的 Markdown 表格
    table_lines = []
    header = (
        f"{'No':>2} │ {'Entry 时间':^16} │ {'Entry 价':^8} │ "
        f"{'Exit 时间':^16} │ {'Exit 价':^8} │ {'Profit%':^8}"
    )
    separator = "───┼──────────────────┼──────────┼──────────────────┼──────────┼──────────"
    table_lines.append(header)
    table_lines.append(separator)
    for idx, row in results_df.iterrows():
        e_ts = row['entry_time'].strftime('%Y-%m-%d %H:%M')
        x_ts = row['exit_time'].strftime('%Y-%m-%d %H:%M')
        e_pr = float(row['entry'])
        x_pr = float(row['exit'])
        p_pct = float(row['profit_pct'])
        line = (
            f"{idx+1:>2} │ {e_ts:^16} │ {e_pr:>8.2f} │ "
            f"{x_ts:^16} │ {x_pr:>8.2f} │ {p_pct:>8.2f}%"
        )
        table_lines.append(line)

    # 计算累计回报
    cap = INITIAL_CAPITAL
    for p_pct in results_df['profit_pct']:
        cap *= (1 + float(p_pct) / 100)
    cum_ret = (cap / INITIAL_CAPITAL - 1) * 100

    summary_line = (
        f"\n总交易笔数：{len(results_df)}  │  累计回报：{cum_ret:.2f}%  │  "
        f"(初始 {INITIAL_CAPITAL:.2f} → 最终 {cap:.2f})"
    )
    history_msg = "📈【历史已完成交易】\n```\n" + "\n".join(table_lines) + "\n```" + summary_line
else:
    history_msg = "📈【历史已完成交易】\n无已完成记录"

# ====== 未平仓交易整理 ======
if open_trades:
    lines = ["📌【未平仓交易】"]
    for idx, ot in enumerate(open_trades, start=1):
        et = pd.to_datetime(ot['entry_time'])
        e_pr = float(ot['entry'])
        reason = ot['reason']
        lines.append(f"{idx}. Entry: {et.strftime('%Y-%m-%d %H:%M')} @ {e_pr:.2f}  状态: {reason}")
    open_msg = "\n".join(lines)
else:
    open_msg = "📌【未平仓交易】\n无"

# ====== 今日信号整理 ======
today_signals = []
today_date = pd.Timestamp.utcnow().tz_convert("UTC").date()

if completed_trades:
    if not pd.api.types.is_datetime64tz_dtype(results_df['entry_time']):
        results_df['entry_time'] = pd.to_datetime(results_df['entry_time']).dt.tz_localize('UTC')
    mask_today = results_df['entry_time'].dt.tz_convert('UTC').dt.date == today_date
    df_today = results_df[mask_today]
    for idx, row in df_today.iterrows():
        e_ts = row['entry_time'].strftime('%Y-%m-%d %H:%M')
        x_ts = row['exit_time'].strftime('%Y-%m-%d %H:%M')
        e_pr = float(row['entry'])
        x_pr = float(row['exit'])
        p_pct = float(row['profit_pct'])
        today_signals.append(
            f"👉 今日已完成 {idx+1}. Entry: {e_ts} @ {e_pr:.2f}  Exit: {x_ts} @ {x_pr:.2f}  Profit: {p_pct:.2f}%"
        )

if not today_signals and open_trades:
    for ot in open_trades:
        et = pd.to_datetime(ot['entry_time'])
        if et.tz_convert('UTC').date() == today_date:
            e_pr = float(ot['entry'])
            today_signals.append(
                f"👉 今日新信号. Entry: {et.strftime('%Y-%m-%d %H:%M')} @ {e_pr:.2f}  状态: {ot['reason']}"
            )

if not today_signals:
    today_signals = ["📊 今日无 W 底信号"]

today_msg = "📅【今日信号】\n" + "\n".join(today_signals)

# ====== 合并最终要发送的文字消息 ======
final_text = "\n\n".join([history_msg, open_msg, today_msg])

# —— coroutine：先发送文字，再（如果有 pattern_points）发送图片 —— #
async def _send_to_telegram(text, chart_path=None):
    # 1. 发送文字
    await bot.send_message(chat_id=CHAT_ID, text=text, parse_mode="Markdown")
    # 2. 如果有图，发送图
    if chart_path and os.path.exists(chart_path):
        with open(chart_path, 'rb') as f:
            await bot.send_photo(chat_id=CHAT_ID, photo=f)

# 调用上面 coroutine
# （先把可能的图保存到本地，再一并发出）
chart_file = None
if pattern_points:
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Close'], color='gray', alpha=0.5, label='Close')
    plotted = set()
    def safe_label(lbl):
        if lbl in plotted:
            return "_nolegend_"
        plotted.add(lbl)
        return lbl

    # 标注已完成交易的进/出场
    for tr in completed_trades:
        ax.scatter(tr['entry_time'], tr['entry'], marker='^', c='green', label=safe_label('Entry'))
        ax.scatter(tr['exit_time'],  tr['exit'],  marker='v', c='red',   label=safe_label('Exit'))

    # 标注 W 底结构
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

    # 保存图到本地文件
    chart_file = "w_pattern_plot.png"
    plt.savefig(chart_file)
    plt.close()

# 运行 coroutine，把文字和（若存在）图片一起发给 Telegram
asyncio.run(_send_to_telegram(final_text, chart_path=chart_file))
