#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import telegram

# ====== 环境变量（请在 GitHub Secrets 或本地环境中设置这两个） ======
# BOT_TOKEN: 你从 @BotFather 那里得到的 Bot 令牌
# CHAT_ID:  你要发送消息的 Telegram 聊天 ID（可以是私聊 ID 或 群组 ID）
BOT_TOKEN ="7231722124:AAHyhoDh04thsN-F3vRxsrxWIDp7WUOzuBk"
CHAT_ID   = "1243234108"

bot = telegram.Bot(token=BOT_TOKEN)

bot.send_message(chat_id=CHAT_ID, text=f"Morning~~")

