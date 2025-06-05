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

# ———— 调试：打印环境变量是否存在 ———— #
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
TICKER = "2330.TW"       # 如果无法下载，可改为大写 "2330.TW"
INTERVAL = "60m"
PERIOD   = "600d"

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
# 注意：yfinance.download() 的 auto_adjust 参数在新版被默认改为 True，如果需要未复权，请显式设置 auto_adjust=False
df = yf.download(TICKER, interval=INTERVAL, period=PERIOD, auto_adjust=False)
if df.empty:
    # 如果下载失败，直接通知并退出
    bot.send_message(chat_id=CHAT_ID, text=f"❌ 无法获取 {TICKER} 的数据，请检查交易所符号或网络。")
    sys.exit(0)

df.dropna(inplace=True)

# 转为 numpy arrays
close_prices = df['Close'].to_numpy()
high_prices  = df['High'].to_numpy()
low_prices   = df['Low'].to_numpy()

# ====== 寻找 W 底信号 ======
pullback_signals = []   # (触发索引, 触发价, 颈线价)
pattern_points   = []   # 保存每个 W 底的各个节点，方便绘图

def detect_w(min_idx, max_idx, tol_p1p3, lo, hi):
    """
    Detect W patterns, 遍历所有相邻的 min_idx（局部极小值）对：
      - p1 = min_idx[i-1], p3 = min_idx[i]
      - p2 必须是 p1~p3 之间的最后一个局部极大值
      - 检查 P1 < P2 且 P3 < P2，且 |P1-P3|/P1 <= tol_p1p3
      - 突破点 bo_i = p3 + 1，拉回点 pb_i = p3 + 3，触发点 tr_i = p3 + 5
      - 满足 bo_v > neckline*(1+BREAKOUT_PCT)，pb_v 在 [lo*neckline, hi*neckline]，tr_v > pb_v
    符合时，把（tr_i, tr_v, neckline）追加到 pullback_signals，并把节点坐标记录到 pattern_points。
    """
    for i in range(1, len(min_idx)):
        p1 = int(min_idx[i - 1])
        p3 = int(min_idx[i])

        # p2 必须是 p1~p3 之间的局部极大值的最后一个
        mids = max_idx[(max_idx > p1) & (max_idx < p3)]
        if mids.size == 0:
            continue
        p2 = int(mids[-1])

        # 取出收盘价
        p1v = float(close_prices[p1].item())
        p2v = float(close_prices[p2].item())
        p3v = float(close_prices[p3].item())

        # 基本形：两头低中间高
        if not (p1v < p2v and p3v < p2v):
            continue

        # P1 与 P3 必须相近
        if abs(p1v - p3v) / p1v > tol_p1p3:
            continue

        neckline = p2v
        bo_i     = p3 + 1
        if bo_i + 4 >= len(close_prices):
            continue

        bo_v = float(close_prices[bo_i].item())       # 突破后第 1 根
        pb_v = float(close_prices[bo_i + 2].item())   # 突破后第 3 根
        tr_v = float(close_prices[bo_i + 4].item())   # 突破后第 5 根

        # 突破条件：突破点 bo_v 必须 > 颈线*(1+BREAKOUT_PCT)
        if bo_v <= neckline * (1 + BREAKOUT_PCT):
            continue

        # 拉回区间：pb_v 必须在 [lo*neckline, hi*neckline]
        if not (neckline * lo < pb_v < neckline * hi):
            continue

        # 触发点 tr_v 必须高于拉回点 pb_v
        if tr_v <= pb_v:
            continue

        # 满足以上条件，记录信号
        pullback_signals.append((bo_i + 4, tr_v, neckline))
        pattern_points.append((p1, p1v, p2, p2v, p3, p3v, bo_i, bo_v, pb_v, tr_v, tol_p1p3))


# —— 找小型 W ——
min_idx_small = argrelextrema(close_prices, np.less_equal, order=MIN_ORDER_SMALL)[0]
max_idx_small = argrelextrema(close_prices, np.greater_equal, order=MIN_ORDER_SMALL)[0]
detect_w(min_idx_small, max_idx_small, P1P3_TOL_SMALL, PULLBACK_LO_SMALL, PULLBACK_HI_SMALL)

# —— 找大型 W ——
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

    # 自 entry_idx+1 开始往后扫，直到触发移动止盈或固定止损
    for offset in range(1, len(df) - entry_idx):
        h = float(high_prices[entry_idx + offset].item())
        l = float(low_prices[entry_idx + offset].item())
        peak = max(peak, h)

        trail_stop = peak * (1 - TRAILING_PCT)     # 移动止盈价
        fixed_stop = entry_price * (1 - STOP_PCT)  # 固定止损价
        stop_level = max(trail_stop, fixed_stop)

        if l <= stop_level:
            # 触发止盈/止损
            result     = 'win' if peak > entry_price else 'loss'
            exit_price = stop_level
            exit_idx   = entry_idx + offset
            break

    # 如果从来没有触发止盈/止损，则最后收盘平仓
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

# ====== 将回测结果转为 DataFrame，并计算 profit_pct ======
if results:
    results_df = pd.DataFrame(results)
    # 确保 entry_time 与 exit_time 都是 datetime
    results_df['entry_time'] = pd.to_datetime(results_df['entry_time'])
    results_df['exit_time']  = pd.to_datetime(results_df['exit_time'])
    results_df['profit_pct'] = (results_df['exit'] - results_df['entry']) / results_df['entry'] * 100
    # 按 entry_time 排序
    results_df.sort_values(by='entry_time', inplace=True)
    results_df.reset_index(drop=True, inplace=True)
else:
    # 如果完全没有信号，创建一个空的 DataFrame（包含所有列，以免下面访问时报错）
    results_df = pd.DataFrame(columns=['entry_time', 'entry', 'exit_time', 'exit', 'result', 'profit_pct'])

# ====== 判断“今天”是否有新信号，并发送 Telegram 消息 ======
today_utc_date = pd.Timestamp.utcnow().date()
results_today  = pd.DataFrame()

if not results_df.empty:
    # 先把 entry_time 都转为 UTC 时区（如果原本不是带时区索引的话）
    # 然后再取出 date 部分进行打平比较
    results_today = results_df.loc[
        results_df['entry_time'].dt.tz_localize('UTC').dt.date == today_utc_date
    ]

# 构造“今日信号”消息
if not results_today.empty:
    lines_today = ["📈 【今日新增 W 底信号】"]
    for idx, row in results_today.iterrows():
        e_time  = row['entry_time'].strftime('%Y-%m-%d %H:%M')
        x_time  = row['exit_time'].strftime('%Y-%m-%d %H:%M')
        e_price = float(row['entry'])
        x_price = float(row['exit'])
        p_pct   = float(row['profit_pct'])
        lines_today.append(
            f"{idx+1}. Entry: {e_time}  @ {e_price:.2f}  →  Exit: {x_time}  @ {x_price:.2f}  Profit: {p_pct:.2f}%"
        )
    # 计算今日单笔累计收益（假设从 INITIAL_CAPITAL 起，只做全部“今日”信号）
    cap_today = INITIAL_CAPITAL
    for p_pct in results_today['profit_pct']:
        cap_today *= (1 + float(p_pct) / 100)
    cum_ret_today = (cap_today / INITIAL_CAPITAL - 1) * 100
    lines_today.append(f"💰 今日交易累计回报：{cum_ret_today:.2f}%")
    text_today = "\n".join(lines_today)
    bot.send_message(chat_id=CHAT_ID, text=text_today)
else:
    # 如果“今天”没有信号
    bot.send_message(chat_id=CHAT_ID, text="📊 今日无 W 底信号。")

# ====== 构造“历史回测结果”消息，并发送 ======
if not results_df.empty:
    # 先把历史所有信号逐条列出
    lines_hist = ["📚 【历史回测结果】"]
    for idx, row in results_df.iterrows():
        e_time  = row['entry_time'].strftime('%Y-%m-%d %H:%M')
        x_time  = row['exit_time'].strftime('%Y-%m-%d %H:%M')
        e_price = float(row['entry'])
        x_price = float(row['exit'])
        p_pct   = float(row['profit_pct'])
        lines_hist.append(
            f"{idx+1}. Entry: {e_time} @ {e_price:.2f}  →  Exit: {x_time} @ {x_price:.2f}  Profit: {p_pct:.2f}%"
        )
    # 然后加一行“从 INITIAL_CAPITAL 到现在的累计回报”
    cap_all = INITIAL_CAPITAL
    for p_pct in results_df['profit_pct']:
        cap_all *= (1 + float(p_pct) / 100)
    cum_ret_all = (cap_all / INITIAL_CAPITAL - 1) * 100
    lines_hist.append(f"\n🔢 初始资金：{INITIAL_CAPITAL:.2f}，当前资金：{cap_all:.2f}，累计回报：{cum_ret_all:.2f}%")

    text_hist = "\n".join(lines_hist)
    bot.send_message(chat_id=CHAT_ID, text=text_hist)
else:
    # 如果历史也没有任何信号
    bot.send_message(chat_id=CHAT_ID, text="⚠️ 历史回测未发现任何 W 底信号。")

# ====== （可选）绘图：仅供本地/调试时参考，不影响 Telegram 推送 ======
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
        ax.scatter(df.index[bo_i+4], tr_v, c='lime',   marker='x', label=safe_label('Trigger'))

    ax.set_title(f"{TICKER} W-Pattern Strategy")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend(loc="best")
    ax.grid(True)
    plt.tight_layout()
    # 如果需要保存图片，可取消下面两行的注释，再让 GitHub Actions 上传 artifact：
    # plt.savefig("w_pattern_plot.png")
    # plt.close()
