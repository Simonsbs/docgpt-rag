import csv
from datetime import datetime
import os

LOG_FILE = "query_history.csv"

def log_query(question: str, response: str):
    is_new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "question", "response"])
        writer.writerow([datetime.now().isoformat(), question, response])
