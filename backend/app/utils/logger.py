from datetime import datetime
from termcolor import colored

def log_step(msg):
    print(colored(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", "cyan"))

def log_success(msg):
    print(colored(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", "green"))

def log_error(msg):
    print(colored(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", "red"))
