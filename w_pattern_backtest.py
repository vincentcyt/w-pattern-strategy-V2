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
# 注意：yfinance.download() 的 auto_adjust 参数在新版被默认改为 True，如果想关闭请显式设置 auto_adjust=False
df = yf.download(TICKER, interval=INTERVAL, period=PERIOD, auto_adjust=False)
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
        p1 = int(min_idx[i - 1])
        p3 = int(min_idx[i])

        # p2 必须是 p1~p3 之间的最高点
        mids = max_idx[(max_idx > p1) & (max_idx < p3)]
        if mids.size == 0:
            continue
        p2 = int(mids[-1])

        # 取出收盘价
        p1v = float(close_prices[p1].item())
        p2v = float(close_prices[p2].item())
        p3v = float(close_prices[p3].item())

        # 基本形态：两头低中间高
        if not (p1v < p2v and p3v < p2v):
            continue

        # P1 与 P3 必须相近
        if abs(p1v - p3v) / p1v > tol_p1p3:
            continue

        neckline = p2v
        bo_i     = p3 + 1
        if bo_i + 4 >= len(close_prices):
            continue

        bo_v = float(close_prices[bo_i].item())
        pb_v = float(close_prices[bo_i + 2].item())
        tr_v = float(close_prices[bo_i + 4].item())

        # 突破条件：突破点必须高于颈线*(1+BREAKOUT_PCT)
        if bo_v <= neckline * (1 + BREAKOUT_PCT):
            continue

        # 拉回区间
        if not (neckline * lo < pb_v < neckline * hi):
            continue

        # 触发点必须高于拉回点
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
results = []
for entry_idx, entry_price, neckline in pullback_signals:
    entry_time = df.index[entry_idx]
    peak       = entry_price
    result     = None
    exit_price = None
    exit_idx   = None

    for offset in range(1, len(df) - entry_idx):
        h = float(high_prices[entry_idx + offset].item())
        l = float(low_prices[entry_idx + offset].item())
        peak = max(peak, h)

        trail_stop = peak * (1 - TRAILING_PCT)
        fixed_stop = entry_price * (1 - STOP_PCT)
        stop_level = max(trail_stop, fixed_stop)

        if l <= stop_level:
            result     = 'win' if peak > entry_price else 'loss'
            exit_price = stop_level
            exit_idx   = entry_idx + offset
            break

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


# ====== 结果展示并推送到 Telegram ======
if results:
    results_df = pd.DataFrame(results)
    results_df['profit_pct'] = (results_df['exit'] - results_df['entry']) / results_df['entry'] * 100

    # —— 构造【历史回测总览】文本 —— #
    total_trades = len(results_df)
    cap = INITIAL_CAPITAL
    for p_pct in results_df['profit_pct']:
        cap *= (1 + float(p_pct) / 100)
    cum_ret = (cap / INITIAL_CAPITAL - 1) * 100

    msg_lines = []
    msg_lines.append("=== 历史回测总览 ===")
    msg_lines.append(f"• 总交易笔数：{total_trades}")
    msg_lines.append(f"• 累计回报率：{cum_ret:.2f}%  (初始资金 {INITIAL_CAPITAL:.2f} → 最终资金 {cap:.2f})")
    msg_lines.append("")

    # —— 把逐笔交易明细以等宽表格的方式展示 —— #
    df_display = results_df.copy()
    df_display['entry_time'] = df_display['entry_time'].dt.strftime('%Y-%m-%d %H:%M')
    df_display['exit_time']  = df_display['exit_time'].dt.strftime('%Y-%m-%d %H:%M')
    df_display['entry']      = df_display['entry'].map(lambda x: f"{float(x):.2f}")
    df_display['exit']       = df_display['exit'].map(lambda x: f"{float(x):.2f}")
    df_display['profit_pct'] = df_display['profit_pct'].map(lambda x: f"{float(x):.2f}%")

    df_display = df_display[[
        'entry_time', 'entry', 'exit_time', 'exit', 'profit_pct'
    ]].rename(columns={
        'entry_time': 'Entry Time',
        'entry':      'Entry',
        'exit_time':  'Exit Time',
        'exit':       'Exit',
        'profit_pct': 'Profit(%)'
    })

    table_text = df_display.to_string(index=False)

    msg_lines.append("=== 逐笔交易明细（等宽表格） ===")
    msg_lines.append("```")
    msg_lines.append(table_text)
    msg_lines.append("```")
else:
    msg_lines = ["⚠️ 历史回测未能检测到任何交易信号。"]

# —— 当日新信号部分 —— #
today_utc_date = pd.Timestamp.utcnow().normalize()
new_today_signals = []
if results:
    for r in results:
        entry_dt_utc = r['entry_time'].tz_convert('UTC').tz_localize(None)
        if entry_dt_utc.date() == today_utc_date.date():
            new_today_signals.append(r)

msg_lines.append("")  # 空行
if new_today_signals:
    msg_lines.append(f"📈 今日新信号：共 {len(new_today_signals)} 笔")
    for idx, r in enumerate(new_today_signals, start=1):
        e_price = float(r['entry'])
        line = (
            f"{idx}. Entry: {r['entry_time'].strftime('%Y-%m-%d %H:%M')} @ {e_price:.2f}  "
            f"(Trigger Time: {r['exit_time'].strftime('%Y-%m-%d %H:%M')})"
        )
        msg_lines.append(line)
else:
    msg_lines.append("📊 今日无 W 底新信号。")

final_msg = "\n".join(msg_lines)

# —— 异步发送给 Telegram —— #
async def _send():
    await bot.send_message(chat_id=CHAT_ID, text=final_msg, parse_mode="Markdown")

asyncio.run(_send())


# ====== （可选）绘图部分，仅供调试时查看结构，不必 GitHub Actions 上传 =====#
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
    # 如果想把图也保存到 artifact，可以解除下面注释并 GitHub Actions 把 w_pattern_plot.png 保留
    # plt.savefig("w_pattern_plot.png")
    # plt.close()
