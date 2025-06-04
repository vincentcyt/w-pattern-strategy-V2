# w_pattern_alert.py
import os
import time
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import telegram

# ------------------ 參數區 ------------------
TICKER = "2330.tw"
INTERVAL = "60m"
PERIOD   = "600d"

# 小型 W 參數
MIN_ORDER_SMALL = 3
P1P3_TOL_SMALL  = 0.9
PULLBACK_LO_SMALL, PULLBACK_HI_SMALL = 0.8, 1.2

# 大型 W 參數
MIN_ORDER_LARGE = 200
P1P3_TOL_LARGE  = 0.9
PULLBACK_LO_LARGE, PULLBACK_HI_LARGE = 0.78, 1.4

# 進出場百分比
BREAKOUT_PCT    = 0.00001
TRAILING_PCT    = 0.08
STOP_PCT        = 0.1

# 初始資金（回測用）
INITIAL_CAPITAL = 100.0

# ------------- 取得 Telegram Bot 參數 -------------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
if BOT_TOKEN is None or CHAT_ID is None:
    raise RuntimeError("請先在環境變數設定 TELEGRAM_BOT_TOKEN 與 TELEGRAM_CHAT_ID")
bot = telegram.Bot(token=BOT_TOKEN)

# ------------- W 底偵測函式 -------------
def detect_w_pattern(close_prices, high_prices, low_prices,
                     min_idx, max_idx, tol_p1p3, lo, hi):
    """
    找出所有滿足 W 底進場條件的訊號。
    回傳兩個 list：
      1. pullback_signals: [(entry_idx, entry_price, neckline), ...]
      2. pattern_points:   [(p1, p1v, p2, p2v, p3, p3v, bo_i, bo_v, pb_v, tr_v, tol), ...]
    """
    pullback_signals = []
    pattern_points   = []
    for i in range(1, len(min_idx)):
        p1 = int(min_idx[i-1])
        p3 = int(min_idx[i])
        # p2 必須是 p1~p3 之間的最大值
        mids = max_idx[(max_idx > p1) & (max_idx < p3)]
        if mids.size == 0:
            continue
        p2 = int(mids[-1])
        p1v = float(close_prices[p1])
        p2v = float(close_prices[p2])
        p3v = float(close_prices[p3])
        # 基本型態：p1 < p2、p3 < p2
        if not (p1v < p2v and p3v < p2v):
            continue
        # P1-P3 高度相似度檢查
        if abs(p1v - p3v) / p1v > tol_p1p3:
            continue
        # 颈线
        neckline = p2v
        bo_i    = p3 + 1
        if bo_i + 4 >= len(close_prices):
            continue
        bo_v = float(close_prices[bo_i])      # 突破收盤
        pb_v = float(close_prices[bo_i + 2])  # 拉回點
        tr_v = float(close_prices[bo_i + 4])  # 觸發點
        # 進場條件
        if bo_v <= neckline * (1 + BREAKOUT_PCT):
            continue
        if not (neckline * lo < pb_v < neckline * hi):
            continue
        if tr_v <= pb_v:
            continue
        # 如果都符合，就加入 signal 列表
        pullback_signals.append((bo_i + 4, tr_v, neckline))
        pattern_points.append((p1, p1v, p2, p2v, p3, p3v,
                               bo_i, bo_v, pb_v, tr_v, tol_p1p3))
    return pullback_signals, pattern_points

# ------------- 主流程 -------------
def run_once():
    # 1. 下載歷史資料
    df = yf.download(TICKER, interval=INTERVAL, period=PERIOD)
    df.dropna(inplace=True)
    close_prices = df['Close'].to_numpy()
    high_prices  = df['High'].to_numpy()
    low_prices   = df['Low'].to_numpy()

    # 2. 找極值索引
    min_idx_small = argrelextrema(close_prices, np.less_equal, order=MIN_ORDER_SMALL)[0]
    max_idx_small = argrelextrema(close_prices, np.greater_equal, order=MIN_ORDER_SMALL)[0]
    min_idx_large = argrelextrema(close_prices, np.less_equal, order=MIN_ORDER_LARGE)[0]
    max_idx_large = argrelextrema(close_prices, np.greater_equal, order=MIN_ORDER_LARGE)[0]

    # 3. 偵測小型 W
    signals_small, patterns_small = detect_w_pattern(
        close_prices, high_prices, low_prices,
        min_idx_small, max_idx_small,
        P1P3_TOL_SMALL, PULLBACK_LO_SMALL, PULLBACK_HI_SMALL
    )
    # 4. 偵測大型 W
    signals_large, patterns_large = detect_w_pattern(
        close_prices, high_prices, low_prices,
        min_idx_large, max_idx_large,
        P1P3_TOL_LARGE, PULLBACK_LO_LARGE, PULLBACK_HI_LARGE
    )

    pullback_signals = signals_small + signals_large
    pattern_points   = patterns_small + patterns_large

    # 5. 如果偵測到進場訊號，就發 Telegram 訊息
    if pullback_signals:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        msg = ["📊 「W 底」偵測到進場訊號：" + now]
        for entry_idx, entry_price, neckline in pullback_signals:
            entry_time = df.index[entry_idx].strftime("%Y-%m-%d %H:%M")
            msg.append(f" • 進場時間：{entry_time}，價位：{entry_price:.2f}，頸線：{neckline:.2f}")
        full_msg = "\n".join(msg)
        bot.send_message(chat_id=CHAT_ID, text=full_msg)
    else:
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "→ 今日無 W 底進場訊號。")

    # 6. 回測：列出每筆交易的 entry/exit、profit_pct，並計算資金演變
    results = []
    for entry_idx, entry_price, neckline in pullback_signals:
        entry_time = df.index[entry_idx]
        peak       = entry_price
        result     = None
        exit_idx   = None
        for j in range(1, len(df) - entry_idx):
            high = float(high_prices[entry_idx + j])
            low  = float(low_prices[entry_idx + j])
            peak = max(peak, high)
            trail_stop = peak * (1 - TRAILING_PCT)
            fixed_stop = entry_price * (1 - STOP_PCT)
            stop_level = max(trail_stop, fixed_stop)
            if low <= stop_level:
                result     = 'win' if peak > entry_price else 'loss'
                exit_price = stop_level
                exit_idx   = entry_idx + j
                break
        # 如果沒碰到止盈或止損，就收盤在最後一根 K
        if result is None:
            exit_idx   = len(df) - 1
            exit_price = float(close_prices[exit_idx])
            result     = 'win' if exit_price > entry_price else 'loss'
        results.append({
            'entry_time': entry_time,
            'entry':      entry_price,
            'exit_time':  df.index[exit_idx],
            'exit':       exit_price,
            'result':     result
        })

    if results:
        results_df = pd.DataFrame(results)
        results_df['profit_pct'] = (results_df['exit'] - results_df['entry']) / results_df['entry'] * 100
        print("\n===== 回測結果 =====")
        print(results_df)
        cap = INITIAL_CAPITAL
        for pct in results_df['profit_pct']:
            cap *= (1 + float(pct)/100)
        cum_ret = (cap/INITIAL_CAPITAL - 1) * 100
        print(f"初始資金 {INITIAL_CAPITAL:.2f} → 最終 {cap:.2f}，累積報酬 {cum_ret:.2f}%\n")
    else:
        print("⚠️ 回測：本次無交易信號。")

    # 7. （選用）畫一張圖，標出進/出場
    #    如果你想在 GitHub Actions 上產生圖片檔，可以把下面這段取消註解，存成 png：
    #
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(12, 5))
    # ax.plot(df['Close'], color='gray', alpha=0.5, label='Close')
    # plotted = set()
    # def safe_label(lbl):
    #     if lbl in plotted: return "_nolegend_"
    #     plotted.add(lbl)
    #     return lbl
    # for tr in results:
    #     ax.scatter(tr['entry_time'], tr['entry'], marker='^', c='green', label=safe_label("Entry"))
    #     ax.scatter(tr['exit_time'], tr['exit'], marker='v', c='red',   label=safe_label("Exit"))
    # ax.set_title(f"{TICKER} W-Pattern Strategy")
    # ax.set_xlabel("Time"); ax.set_ylabel("Price")
    # ax.legend(loc="best"); ax.grid(True); plt.tight_layout()
    # plt.savefig("w_pattern_chart.png", dpi=150)
    # plt.close(fig)


if __name__ == "__main__":
    run_once()
