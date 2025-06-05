import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from telegram import Bot

# ====== 环境变量读取 ======
BOT_TOKEN = os.getenv("BOT_TOKEN")    # 由 GitHub Secrets 注入
CHAT_ID   = os.getenv("CHAT_ID")      # 由 GitHub Secrets 注入

if not BOT_TOKEN or not CHAT_ID:
    raise RuntimeError("请先在环境变量中设置 BOT_TOKEN 和 CHAT_ID。")

bot = Bot(token=BOT_TOKEN)

# ====== 参数区（方便调整） ======
TICKER = "2330.tw"
INTERVAL = "60m"        # 数据周期
PERIOD   = "600d"       # 数据长度

# 小型 W 参数
MIN_ORDER_SMALL      = 3       # 小型 W 极值识别窗口
P1P3_TOL_SMALL       = 0.9     # P1 与 P3 相似度容差（小型 W）
PULLBACK_LO_SMALL    = 0.8     # 小型 W 拉回区域下限
PULLBACK_HI_SMALL    = 1.2     # 小型 W 拉回区域上限

# 大型 W 参数
MIN_ORDER_LARGE      = 200     # 大型 W 极值识别窗口 (约一天以上周期)
P1P3_TOL_LARGE       = 0.9     # P1 与 P3 相似度容差（大型 W）
PULLBACK_LO_LARGE    = 0.78    # 大型 W 拉回区域下限（放宽）
PULLBACK_HI_LARGE    = 1.4     # 大型 W 拉回区域上限（放宽)

# 统一参数
BREAKOUT_PCT    = 0.00001     # 突破颈线百分比
INITIAL_CAPITAL = 100.0       # 初始资金
TRAILING_PCT    = 0.08        # 移动止盈百分比
STOP_PCT        = 0.1         # 固定止损百分比

# ====== 数据下载 ======
# 强制将 auto_adjust 设为 False，以避免默认值变更带来的潜在问题
df = yf.download(TICKER, interval=INTERVAL, period=PERIOD, auto_adjust=False)
df.dropna(inplace=True)

# 转为 numpy arrays
close_prices = df['Close'].to_numpy()
high_prices  = df['High'].to_numpy()
low_prices   = df['Low'].to_numpy()

# ====== 寻找 W 底信号 ======
pullback_signals = []   # (entry_idx, entry_price, neckline)
pattern_points   = []   # (p1, p1v, p2, p2v, p3, p3v, bo_i, bo_v, pb_v, tr_v, tol)

def detect_w(min_idx, max_idx, tol_p1p3, lo, hi):
    """
    Detect W patterns given extrema indices and tolerances.
    lo/hi define pullback zone multipliers for neckline.
    """
    for i in range(1, len(min_idx)):
        p1 = int(min_idx[i-1])
        p3 = int(min_idx[i])
        # p2 must be highest点 between p1 and p3
        mids = max_idx[(max_idx > p1) & (max_idx < p3)]
        if mids.size == 0:
            continue
        p2 = int(mids[-1])
        # extract values as python floats
        p1v = close_prices[p1].item()
        p2v = close_prices[p2].item()
        p3v = close_prices[p3].item()
        # 基本形态检查：P1,P3 都要低于 P2
        if not (p1v < p2v and p3v < p2v):
            continue
        # P1-P3 相似度检查
        if abs(p1v - p3v) / p1v > tol_p1p3:
            continue
        # 颈线定义为 p2v
        neckline = p2v
        # 突破点 bo_i 紧跟在 p3 之后
        bo_i = p3 + 1
        if bo_i + 4 >= len(close_prices):
            continue
        bo_v = close_prices[bo_i].item()
        pb_v = close_prices[bo_i + 2].item()
        tr_v = close_prices[bo_i + 4].item()
        # 进场条件
        # 1. 突破点必须超过颈线 * (1 + BREAKOUT_PCT)
        if bo_v <= neckline * (1 + BREAKOUT_PCT):
            continue
        # 2. 回测（pullback）价要落在 [neckline * lo, neckline * hi]
        if not (neckline * lo < pb_v < neckline * hi):
            continue
        # 3. 确认触发点 tr_v 必须高于回测点 pb_v
        if tr_v <= pb_v:
            continue
        # 如果都通过，则记录这个信号
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
results = []  # 存放每笔交易的字典
for entry_idx, entry_price, neckline in pullback_signals:
    entry_time = df.index[entry_idx]
    peak       = entry_price
    result     = None
    exit_idx   = None
    exit_price = None
    # 持有期直到触发止盈/止损
    for j in range(1, len(df) - entry_idx):
        high = high_prices[entry_idx + j].item()
        low  = low_prices[entry_idx + j].item()
        if high > peak:
            peak = high
        trail_stop = peak * (1 - TRAILING_PCT)
        fixed_stop = entry_price * (1 - STOP_PCT)
        stop_level = max(trail_stop, fixed_stop)
        if low <= stop_level:
            # 触发止损或止盈
            result     = 'win' if peak > entry_price else 'loss'
            exit_price = stop_level
            exit_idx   = entry_idx + j
            break
    # 如果没触发，则以当期收盘价作平仓
    if result is None:
        exit_idx   = len(df) - 1
        exit_price = close_prices[exit_idx].item()
        result     = 'win' if exit_price > entry_price else 'loss'
    results.append({
        'entry_time': entry_time,
        'entry':      float(entry_price),
        'exit_time':  df.index[exit_idx],
        'exit':       float(exit_price),
        'result':     result
    })

# ====== 结果展示 ======
if len(results) > 0:
    results_df = pd.DataFrame(results)
    # 计算每笔交易的收益率
    results_df['profit_pct'] = (results_df['exit'] - results_df['entry']) / results_df['entry'] * 100

    # 输出到控制台
    print("每笔交易详情：")
    print(results_df[['entry_time', 'entry', 'exit_time', 'exit', 'result', 'profit_pct']])

    # 计算累积资金
    cap = INITIAL_CAPITAL
    for pct in results_df['profit_pct']:
        cap *= (1 + float(pct) / 100)
    cum_ret = (cap / INITIAL_CAPITAL - 1) * 100

    print(f"\n初始资金：{INITIAL_CAPITAL:.2f} 元，最终资金：{cap:.2f} 元，累计收益：{cum_ret:.2f}%\n")
else:
    print(f"⚠️ 无交易信号，共 {len(pullback_signals)} 个信号")

# ====== 将信号发到 Telegram ======
# ====== 将信号发到 Telegram ======
if len(results) > 0:
    msg = f"📊 {TICKER} W 底策略回测结果：\n"
    for idx, row in results_df.iterrows():
        # 先把 timestamp 转成字符串
        entry_t_str = row['entry_time'].strftime('%Y-%m-%d %H:%M')
        exit_t_str  = row['exit_time'].strftime('%Y-%m-%d %H:%M')
        # 再把数值先转成 float，才能用 {:.2f}
        entry_p     = float(row['entry'])
        exit_p      = float(row['exit'])
        profit_pct  = float(row['profit_pct'])
        msg += (
            f"{idx+1}. Entry: {entry_t_str} @ {entry_p:.2f}，"
            f"Exit: {exit_t_str} @ {exit_p:.2f}，"
            f"Profit: {profit_pct:.2f}%\n"
        )
    msg += f"\n初始 {INITIAL_CAPITAL:.2f}，最终 {cap:.2f}，累计 {cum_ret:.2f}%"
    bot.send_message(chat_id=CHAT_ID, text=msg)

# ====== 绘图 ======
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(df['Close'], color='gray', alpha=0.5, label='Close')
plotted = set()
def safe_label(lbl):
    if lbl in plotted:
        return '_nolegend_'
    plotted.add(lbl)
    return lbl

# 标注进/出场
for tr in results:
    ax.scatter(tr['entry_time'], tr['entry'], marker='^', c='green', label=safe_label('Entry'))
    ax.scatter(tr['exit_time'],  tr['exit'],   marker='v', c='red',   label=safe_label('Exit'))

# 标注结构点（如果需要，可以在这里把 pattern_points 也画出来）
# for p1, p1v, p2, p2v, p3, p3v, bo_i, bo_v, pb_v, tr_v, tol in pattern_points:
#     ax.scatter(df.index[p1], p1v, c='blue', marker='o', label=safe_label('P1'))
#     ax.scatter(df.index[p3], p3v, c='blue', marker='o', label=safe_label('P3'))
#     ax.scatter(df.index[p2], p2v, c='orange', marker='o', label=safe_label('P2'))
#     ax.hlines(p2v, df.index[p1], df.index[p3], colors='purple', linestyles='dashed', label=safe_label('Neckline'))
#     ax.scatter(df.index[bo_i],    bo_v, c='cyan',    marker='x', label=safe_label('Breakout'))
#     ax.scatter(df.index[bo_i+2],  pb_v, c='magenta', marker='x', label=safe_label('Pullback'))
#     ax.scatter(df.index[bo_i+4],  tr_v, c='lime',    marker='x', label=safe_label('Trigger'))

ax.set_title(f"{TICKER} W-Pattern Strategy")
ax.set_xlabel('Time')
ax.set_ylabel('Price')
ax.legend(loc='best')
ax.grid(True)
plt.tight_layout()
plt.savefig("w_pattern_backtest_plot.png")  # 保存为图片
plt.close()

# 如果你还想把图发到 Telegram，可以打开下面两行
# with open("w_pattern_backtest_plot.png", "rb") as photo:
#     bot.send_photo(chat_id=CHAT_ID, photo=photo)
