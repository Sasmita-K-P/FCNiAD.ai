# app/utils/logger.py
from termcolor import colored
from datetime import datetime

def _log(msg, color=None, symbol="ℹ️"):
    now = datetime.now().strftime("%H:%M:%S")
    text = f"[{now}] {symbol} {msg}"
    if color:
        print(colored(text, color))
    else:
        print(text)

def info(msg): _log(msg, "cyan", "ℹ️")
def success(msg): _log(msg, "green", "✅")
def warn(msg): _log(msg, "yellow", "⚠️")
def error(msg): _log(msg, "red", "❌")
