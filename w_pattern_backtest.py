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

# —— 【调试】打印环境变量是否存在 —— #
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = os.getenv("CHAT_ID")
print(f"[DEBUG] BOT_TOKEN is [{'set' if BOT_TOKEN else 'NOT set'}]")
print(f"[DEBUG] CHAT_ID   is [{'set' if CHAT_ID else 'NOT set'}]")

if not BOT_TOKEN or not CHAT_ID:
    print("❌ ERROR: 必须在环境变量里设置 BOT_TOKEN 和 CHAT_ID，程序退出。")
    sys.exit(1)

# 初始化 Telegram Bot
bot = Bot(token=BOT_TOKEN)


# ====== 参数区（方便调整） ======
TICKER = "2330.tw"
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
# 注意：yfinance.download() 的 auto_adjust 参数在新版默认已经改为 True，
# 若要使用历史未复权价格，请显式写 auto_adjust=False
df = yf.download(TICKER, interval=INTERVAL, period=PERIOD, auto_adjust=False)
df.dropna(inplace=True)

# 将 Close/High/Low 转成 numpy arrays，方便快速索引
close_prices = df['Close'].to_numpy()
high_prices  = df['High'].to_numpy()
low_prices   = df['Low'].to_numpy()


# ====== 寻找 W 底信号 ======
pullback_signals = []   # 存储所有检测到的（触发索引、触发价格、颈线价格）
pattern_points   = []   # 存储形态细节，用于画图参考

def detect_w(min_idx, max_idx, tol_p1p3, lo, hi):
    """
    根据局部极值索引和容差范围，找出所有符合 W 底形态的信号点。
    min_idx: 所有局部底（P1/P3）的索引数组
    max_idx: 所有局部顶（P2）的索引数组
    tol_p1p3: P1 与 P3 相似度容差
    lo, hi: 拉回区域对颈线价的乘数范围
    """
    for i in range(1, len(min_idx)):
        p1 = int(min_idx[i - 1])
        p3 = int(min_idx[i])
        # 在 p1 与 p3 之间，寻找最后一个局部顶作为 p2
        mids = max_idx[(max_idx > p1) & (max_idx < p3)]
        if mids.size == 0:
            continue
        p2 = int(mids[-1])
        # 读出具体收盘价
        p1v = float(close_prices[p1].item())
        p2v = float(close_prices[p2].item())
        p3v = float(close_prices[p3].item())
        # 必须满足“两头低中间高”
        if not (p1v < p2v and p3v < p2v):
            continue
        # P1 与 P3 价格要在 tol_p1p3 的范围内
        if abs(p1v - p3v) / p1v > tol_p1p3:
            continue
        # 颈线价格
        neckline = p2v
        # 突破点索引（p3 之后紧接一个 bar 视为突破点）
        bo_i = p3 + 1
        if bo_i + 4 >= len(close_prices):
            # 不足 4 根 K 线来验证拉回+触发，就跳过
            continue
        bo_v = float(close_prices[bo_i].item())       # 突破后的马上一个 bar
        pb_v = float(close_prices[bo_i + 2].item())   # 突破后隔两根 bar
        tr_v = float(close_prices[bo_i + 4].item())   # 触发点：突破后隔四根 bar

        # 进场条件：突破点必须 > 颈线*(1+BROKEOUT_PCT)
        if bo_v <= neckline * (1 + BREAKOUT_PCT):
            continue
        # 拉回必须在 [lo * 颈线, hi * 颈线]
        if not (neckline * lo < pb_v < neckline * hi):
            continue
        # 触发点必须高于拉回点
        if tr_v <= pb_v:
            continue

        # 如果所有条件都满足，就把触发时刻（bo_i+4）加入信号列表
        pullback_signals.append((bo_i + 4, tr_v, neckline))
        # 同时记录下 p1/p2/p3/bo_i 的索引和价格，用于后面画图
        pattern_points.append((
            p1, p1v, p2, p2v, p3, p3v,
            bo_i, bo_v, pb_v, tr_v, tol_p1p3
        ))


# —— 找小型 W —— #
min_idx_small = argrelextrema(close_prices, np.less_equal, order=MIN_ORDER_SMALL)[0]
max_idx_small = argrelextrema(close_prices, np.greater_equal, order=MIN_ORDER_SMALL)[0]
detect_w(min_idx_small, max_idx_small, P1P3_TOL_SMALL, PULLBACK_LO_SMALL, PULLBACK_HI_SMALL)

# —— 找大型 W —— #
min_idx_large = argrelextrema(close_prices, np.less_equal, order=MIN_ORDER_LARGE)[0]
max_idx_large = argrelextrema(close_prices, np.greater_equal, order=MIN_ORDER_LARGE)[0]
detect_w(min_idx_large, max_idx_large, P1P3_TOL_LARGE, PULLBACK_LO_LARGE, PULLBACK_HI_LARGE)


# ====== 回测部分 ======
# pullback_signals 中每一项：(entry_idx, entry_price, neckline)
results = []
for entry_idx, entry_price, neckline in pullback_signals:
    entry_time = df.index[entry_idx]
    peak       = entry_price    # 用于计算移动止盈
    result     = None
    exit_price = None
    exit_idx   = None

    # 从 entry_idx 开始，逐根 bar 判断是否触发止盈或止损
    for offset in range(1, len(df) - entry_idx):
        high = float(high_prices[entry_idx + offset].item())
        low  = float(low_prices[entry_idx + offset].item())
        peak = max(peak, high)

        trail_stop = peak * (1 - TRAILING_PCT)            # 移动止盈价
        fixed_stop = entry_price * (1 - STOP_PCT)         # 固定止损价
        stop_level = max(trail_stop, fixed_stop)          # 以最高者为实际止损止盈价

        if low <= stop_level:
            # 触发止盈/止损
            result     = 'win' if peak > entry_price else 'loss'
            exit_price = stop_level
            exit_idx   = entry_idx + offset
            break

    # 如果持有期结束都没触发止盈/止损，则收盘平仓
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


# ====== 将所有信号汇总为 DataFrame ======
if results:
    results_df = pd.DataFrame(results)
    # 计算每笔交易的收益率
    results_df['profit_pct'] = (results_df['exit'] - results_df['entry']) / results_df['entry'] * 100
    # 按 entry_time 升序排列（通常是默认顺序）
    results_df.sort_values(by='entry_time', inplace=True)
    results_df.reset_index(drop=True, inplace=True)
else:
    results_df = pd.DataFrame(columns=['entry_time','entry','exit_time','exit','result','profit_pct'])


# ====== 判断“今天”信号，并发送 Telegram 消息 ======
# 取当前 UTC 日期作为“今天”的判断标准
# 如果你的运行环境不是 UTC，请相应调整时区。
today_utc_date = pd.Timestamp.utcnow().date()

# 从 results_df 里筛选出 entry_time 属于今天的信号
# 注意：df.index 上包含时区信息，这里我们直接取 date 部分比较
results_today = results_df.loc[
    results_df['entry_time'].dt.tz_convert('UTC').dt.date == today_utc_date
]

if not results_today.empty:
    # 当天有信号，就把所有当天信号都发送
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

    # 同时附上当日到目前为止的累计收益（假设从第一笔开始资金 100 推算）
    cap = INITIAL_CAPITAL
    for p_pct in results_today['profit_pct']:
        cap *= (1 + float(p_pct)/100)
    cum_ret_today = (cap / INITIAL_CAPITAL - 1) * 100
    msg_lines.append(f"💰 今日交易累计回报：{cum_ret_today:.2f}%")

    final_msg = "\n".join(msg_lines)
    bot.send_message(chat_id=CHAT_ID, text=final_msg)

else:
    # 当天没有新增信号，则先发送“今日无信号”，然后把历史上最后一次的信号发送出来
    bot.send_message(chat_id=CHAT_ID, text="📊 今日无 W 底信号，以下为历史上最后一次信号：")

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
        # 历史里也没信号的话，就发送提示
        bot.send_message(chat_id=CHAT_ID, text="⚠️ 历史数据里也没有任何 W 底信号。")



# ====== （可选）画图部分，仅供调试/本地运行时使用，GitHub Actions 无需保存图片 —— #
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

    # 如果你想把图存下来并上传成 GitHub Actions artifact，可以取消下面两行注释并在 workflow 里做相应配置：
    # plt.savefig("w_pattern_plot.png")
    # plt.close()
