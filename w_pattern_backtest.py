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

    # 标注已完成交易的进/出场
    for tr in completed_trades:
        ax.scatter(
            tr['entry_time'], tr['entry'],
            marker='^', c='green', s=50,
            label=safe_label('已完成 Entry')
        )
        ax.scatter(
            tr['exit_time'], tr['exit'],
            marker='v', c='red', s=50,
            label=safe_label('已完成 Exit')
        )

    # 标注尚未平仓交易的进场点（黄色）
    for ot in open_trades:
        ax.scatter(
            ot['entry_time'], ot['entry'],
            marker='^', c='yellow', s=70,
            edgecolors='black', label=safe_label('未平仓 Entry')
        )


        

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
